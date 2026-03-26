/*
 * Vulkan device filter - acts as a thin wrapper around the real libvulkan.
 *
 * When placed on LD_LIBRARY_PATH as "libvulkan.so.1", Dawn will dlopen this
 * instead of the real Vulkan loader. We forward all calls to the real loader
 * but intercept vkEnumeratePhysicalDevices to filter to a single device.
 *
 * Build:
 *   gcc -shared -fPIC -o libvulkan.so.1 vk-device-filter.c -ldl
 *
 * Usage:
 *   VULKAN_DEVICE_INDEX=7 LD_LIBRARY_PATH=/path/to/dir:$LD_LIBRARY_PATH npx tsx ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Minimal Vulkan types */
typedef unsigned int VkResult;
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void (*PFN_vkVoidFunction)(void);

#define VK_SUCCESS 0
#define VK_INCOMPLETE 5

/* Real Vulkan loader handle */
static void* real_vulkan = NULL;

/* Function pointer types */
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, unsigned int*, VkPhysicalDevice*);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(VkInstance, const char*);

static PFN_vkGetInstanceProcAddr real_getInstanceProcAddr = NULL;
static int target_device_index = -1;
static int initialized = 0;

static void ensure_init(void) {
    if (initialized) return;
    initialized = 1;

    /* Load the real Vulkan loader from the standard paths, skipping ourselves */
    /* Try known system paths */
    const char* paths[] = {
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        real_vulkan = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (real_vulkan) break;
    }

    if (!real_vulkan) {
        fprintf(stderr, "[vk-device-filter] ERROR: Cannot find real libvulkan.so.1\n");
        return;
    }

    real_getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(real_vulkan, "vkGetInstanceProcAddr");
    if (!real_getInstanceProcAddr) {
        fprintf(stderr, "[vk-device-filter] ERROR: Cannot find vkGetInstanceProcAddr in real loader\n");
    }

    const char* idx_str = getenv("VULKAN_DEVICE_INDEX");
    if (idx_str) {
        target_device_index = atoi(idx_str);
        fprintf(stderr, "[vk-device-filter] Filtering to device index %d\n", target_device_index);
    }
}

/* Our intercepted vkEnumeratePhysicalDevices */
static VkResult wrapped_EnumeratePhysicalDevices(
    VkInstance instance,
    unsigned int* pPhysicalDeviceCount,
    VkPhysicalDevice* pPhysicalDevices
) {
    ensure_init();

    if (!real_getInstanceProcAddr) return VK_INCOMPLETE;

    PFN_vkEnumeratePhysicalDevices real_enum =
        (PFN_vkEnumeratePhysicalDevices)real_getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    if (!real_enum) return VK_INCOMPLETE;

    if (target_device_index < 0) {
        return real_enum(instance, pPhysicalDeviceCount, pPhysicalDevices);
    }

    /* Get all devices */
    unsigned int total = 0;
    VkResult res = real_enum(instance, &total, NULL);
    if (res != VK_SUCCESS) return res;

    int idx = target_device_index;
    if ((unsigned int)idx >= total) {
        fprintf(stderr, "[vk-device-filter] WARNING: index %d >= total %u, using 0\n", idx, total);
        idx = 0;
    }

    if (!pPhysicalDevices) {
        *pPhysicalDeviceCount = 1;
        return VK_SUCCESS;
    }

    VkPhysicalDevice* all = (VkPhysicalDevice*)malloc(total * sizeof(VkPhysicalDevice));
    if (!all) return VK_INCOMPLETE;
    res = real_enum(instance, &total, all);
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
        free(all);
        return res;
    }

    if (*pPhysicalDeviceCount < 1) {
        free(all);
        *pPhysicalDeviceCount = 1;
        return VK_INCOMPLETE;
    }

    pPhysicalDevices[0] = all[idx];
    *pPhysicalDeviceCount = 1;
    free(all);
    return VK_SUCCESS;
}

/* Main export: vkGetInstanceProcAddr */
PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    ensure_init();

    /* Intercept vkEnumeratePhysicalDevices */
    if (pName && strcmp(pName, "vkEnumeratePhysicalDevices") == 0) {
        return (PFN_vkVoidFunction)wrapped_EnumeratePhysicalDevices;
    }

    /* Forward everything else to real loader */
    if (real_getInstanceProcAddr) {
        return real_getInstanceProcAddr(instance, pName);
    }

    return NULL;
}

/* Also export some commonly needed functions that Dawn might dlsym directly */
typedef VkResult (*PFN_vkCreateInstance)(const void*, const void*, VkInstance*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef VkResult (*PFN_vkEnumerateInstanceVersion)(unsigned int*);
typedef VkResult (*PFN_vkEnumerateInstanceExtensionProperties)(const char*, unsigned int*, void*);
typedef VkResult (*PFN_vkEnumerateInstanceLayerProperties)(unsigned int*, void*);

VkResult vkCreateInstance(const void* pCreateInfo, const void* pAllocator, VkInstance* pInstance) {
    ensure_init();
    PFN_vkCreateInstance fn = (PFN_vkCreateInstance)real_getInstanceProcAddr(NULL, "vkCreateInstance");
    if (fn) return fn(pCreateInfo, pAllocator, pInstance);
    return VK_INCOMPLETE;
}

VkResult vkEnumerateInstanceVersion(unsigned int* pApiVersion) {
    ensure_init();
    PFN_vkEnumerateInstanceVersion fn = (PFN_vkEnumerateInstanceVersion)dlsym(real_vulkan, "vkEnumerateInstanceVersion");
    if (fn) return fn(pApiVersion);
    return VK_INCOMPLETE;
}

VkResult vkEnumerateInstanceExtensionProperties(const char* pLayerName, unsigned int* pPropertyCount, void* pProperties) {
    ensure_init();
    PFN_vkEnumerateInstanceExtensionProperties fn = (PFN_vkEnumerateInstanceExtensionProperties)real_getInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
    if (fn) return fn(pLayerName, pPropertyCount, pProperties);
    return VK_INCOMPLETE;
}

VkResult vkEnumerateInstanceLayerProperties(unsigned int* pPropertyCount, void* pProperties) {
    ensure_init();
    PFN_vkEnumerateInstanceLayerProperties fn = (PFN_vkEnumerateInstanceLayerProperties)real_getInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties");
    if (fn) return fn(pPropertyCount, pProperties);
    return VK_INCOMPLETE;
}

VkResult vkEnumeratePhysicalDevices(VkInstance instance, unsigned int* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) {
    return wrapped_EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
}
