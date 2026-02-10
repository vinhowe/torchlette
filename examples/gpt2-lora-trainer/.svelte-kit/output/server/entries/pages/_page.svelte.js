import { w as attr_class, x as attr, y as ensure_array_like, z as stringify, F as attr_style } from "../../chunks/index.js";
import { e as escape_html } from "../../chunks/context.js";
let isLoaded = false;
let webgpuSupported = null;
let loraRank = 8;
let loraAlpha = 16;
const modelStore = {
  get isLoaded() {
    return isLoaded;
  },
  get webgpuSupported() {
    return webgpuSupported;
  },
  get loraRank() {
    return loraRank;
  },
  get loraAlpha() {
    return loraAlpha;
  },
  // Setters
  set loraRank(v) {
    loraRank = v;
  },
  set loraAlpha(v) {
    loraAlpha = v;
  }
};
let files = [];
let maxSteps = 50;
let batchSize = 1;
let seqLength = 32;
let learningRate = 1e-4;
let isTraining = false;
let currentStep = 0;
let lossHistory = [];
const totalTokens = files.reduce((sum, f) => sum + f.tokens, 0);
files.length > 0 && !isTraining && modelStore.isLoaded;
const trainingStore = {
  // Getters
  get files() {
    return files;
  },
  get maxSteps() {
    return maxSteps;
  },
  get batchSize() {
    return batchSize;
  },
  get seqLength() {
    return seqLength;
  },
  get learningRate() {
    return learningRate;
  },
  get isTraining() {
    return isTraining;
  },
  get currentStep() {
    return currentStep;
  },
  get lossHistory() {
    return lossHistory;
  },
  get totalTokens() {
    return totalTokens;
  },
  // Setters
  set maxSteps(v) {
    maxSteps = v;
  },
  set batchSize(v) {
    batchSize = v;
  },
  set seqLength(v) {
    seqLength = v;
  },
  set learningRate(v) {
    learningRate = v;
  }
};
function FileDropZone($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="bg-slate-800 rounded-lg p-4"><h3 class="text-lg font-semibold text-white mb-3">Training Data</h3> <button${attr_class(`w-full border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${stringify("border-slate-600 hover:border-slate-500")}`)}${attr("disabled", true, true)}><input type="file" accept=".txt,text/plain" multiple class="hidden"/> <svg class="w-10 h-10 mx-auto mb-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg> `);
    {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<p class="text-slate-500">Load the model first to add training data</p>`);
    }
    $$renderer2.push(`<!--]--></button> `);
    if (trainingStore.files.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="mt-4 space-y-2"><!--[-->`);
      const each_array = ensure_array_like(trainingStore.files);
      for (let index = 0, $$length = each_array.length; index < $$length; index++) {
        let file = each_array[index];
        $$renderer2.push(`<div class="flex items-center justify-between bg-slate-700 rounded-md px-3 py-2"><div class="flex items-center gap-2"><svg class="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"></path></svg> <span class="text-slate-200 text-sm">${escape_html(file.name)}</span> <span class="text-slate-500 text-xs">(${escape_html(file.tokens.toLocaleString())} tokens)</span></div> <button class="text-slate-400 hover:text-red-400 transition-colors"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg></button></div>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="mt-3 flex justify-between items-center"><span class="text-slate-400 text-sm">Total: ${escape_html(trainingStore.totalTokens.toLocaleString())} tokens</span> <button class="text-sm text-slate-400 hover:text-slate-200 transition-colors">Clear all</button></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function TrainingProgress($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const chartWidth = 300;
    const chartHeight = 100;
    const chartPath = () => {
      const history = trainingStore.lossHistory;
      if (history.length < 2) return "";
      const maxLoss = Math.max(...history, 0.1);
      const minLoss = Math.min(...history, 0);
      const range = maxLoss - minLoss || 1;
      const points = history.map((loss, i) => {
        const x = i / (history.length - 1) * chartWidth;
        const y = chartHeight - (loss - minLoss) / range * chartHeight;
        return `${x},${y}`;
      });
      return `M ${points.join(" L ")}`;
    };
    const progressPercent = trainingStore.maxSteps > 0 ? trainingStore.currentStep / trainingStore.maxSteps * 100 : 0;
    $$renderer2.push(`<div class="bg-slate-800 rounded-lg p-4"><div class="flex items-center justify-between mb-4"><h3 class="text-lg font-semibold text-white">Training Progress</h3> `);
    {
      $$renderer2.push("<!--[!-->");
      {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div> <div class="mb-4"><div class="flex justify-between text-sm text-slate-400 mb-1"><span>Step ${escape_html(trainingStore.currentStep)} / ${escape_html(trainingStore.maxSteps)}</span> <span>${escape_html(progressPercent.toFixed(0))}%</span></div> <div class="w-full bg-slate-700 rounded-full h-2"><div class="bg-green-500 h-2 rounded-full transition-all duration-300"${attr_style(`width: ${stringify(progressPercent)}%`)}></div></div></div> <div class="grid grid-cols-2 gap-4 mb-4"><div class="bg-slate-700 rounded-md p-3"><p class="text-slate-400 text-xs uppercase mb-1">Loss</p> <p class="text-white text-lg font-mono">${escape_html("—")}</p></div> <div class="bg-slate-700 rounded-md p-3"><p class="text-slate-400 text-xs uppercase mb-1">Tokens/sec</p> <p class="text-white text-lg font-mono">${escape_html("—")}</p></div></div> `);
    if (trainingStore.lossHistory.length > 1) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="bg-slate-700 rounded-md p-3"><p class="text-slate-400 text-xs uppercase mb-2">Loss History</p> <svg${attr("viewBox", `0 0 ${stringify(chartWidth)} ${stringify(chartHeight)}`)} class="w-full h-24" preserveAspectRatio="none"><path${attr("d", chartPath())} fill="none" stroke="#22c55e" stroke-width="2"></path></svg></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="mt-4 p-3 bg-slate-700 rounded-md"><p class="text-slate-400 text-sm">`);
      if (trainingStore.files.length === 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`Add training files to begin`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (trainingStore.totalTokens < trainingStore.batchSize * (trainingStore.seqLength + 1)) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`Need at least ${escape_html(trainingStore.batchSize * (trainingStore.seqLength + 1))} tokens
          (have ${escape_html(trainingStore.totalTokens)})`);
        } else {
          $$renderer2.push("<!--[!-->");
          $$renderer2.push(`Load the model first`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></p></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function LoRAConfig($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="bg-slate-800 rounded-lg p-4"><h3 class="text-lg font-semibold text-white mb-4">Training Configuration</h3> <div class="space-y-4"><div><label class="block text-sm text-slate-300 mb-1">LoRA Rank: ${escape_html(modelStore.loraRank)}</label> <input type="range" min="2" max="32" step="2"${attr("value", modelStore.loraRank)} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/> <p class="text-xs text-slate-500 mt-1">Lower = smaller adapter, higher = more capacity</p></div> <div><label class="block text-sm text-slate-300 mb-1">LoRA Alpha: ${escape_html(modelStore.loraAlpha)}</label> <input type="range" min="4" max="64" step="4"${attr("value", modelStore.loraAlpha)} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/> <p class="text-xs text-slate-500 mt-1">Scaling factor (typically equal to rank)</p></div> <hr class="border-slate-700"/> <div><label class="block text-sm text-slate-300 mb-1">Training Steps: ${escape_html(trainingStore.maxSteps)}</label> <input type="range" min="10" max="500" step="10"${attr("value", trainingStore.maxSteps)} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/></div> <div><label class="block text-sm text-slate-300 mb-1">Batch Size: ${escape_html(trainingStore.batchSize)}</label> <input type="range" min="1" max="4" step="1"${attr("value", trainingStore.batchSize)} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/> <p class="text-xs text-slate-500 mt-1">Keep low to reduce GPU memory usage</p></div> <div><label class="block text-sm text-slate-300 mb-1">Sequence Length: ${escape_html(trainingStore.seqLength)}</label> <input type="range" min="16" max="128" step="16"${attr("value", trainingStore.seqLength)} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/> <p class="text-xs text-slate-500 mt-1">Shorter = less memory, longer = better context</p></div> <div><label class="block text-sm text-slate-300 mb-1">Learning Rate: ${escape_html(trainingStore.learningRate.toExponential(0))}</label> <input type="range" min="-5" max="-3" step="0.5"${attr("value", Math.log10(trainingStore.learningRate))} class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"${attr("disabled", trainingStore.isTraining, true)}/></div> <hr class="border-slate-700"/> <div class="space-y-3"><h4 class="text-sm font-medium text-slate-400">Memory Optimization</h4> <div class="flex items-center justify-between"><div><span class="text-sm text-slate-300">Mixed Precision (AMP)</span> <p class="text-xs text-slate-500">Use f16 for compute, reduces memory ~50%</p></div> <button${attr_class(`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${stringify("bg-blue-600")}`)}${attr("disabled", trainingStore.isTraining, true)}><span${attr_class(`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${stringify("translate-x-6")}`)}></span></button></div> <div class="flex items-center justify-between"><div><span class="text-sm text-slate-300">Gradient Checkpointing</span> <p class="text-xs text-slate-500">Trade compute for memory, ~2x slower</p></div> <button${attr_class(`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${stringify("bg-blue-600")}`)}${attr("disabled", trainingStore.isTraining, true)}><span${attr_class(`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${stringify("translate-x-6")}`)}></span></button></div></div></div></div>`);
  });
}
let messages = [];
let temperature = 0.7;
let maxTokens = 100;
let topK = 50;
const chatStore = {
  // Getters
  get messages() {
    return messages;
  },
  get temperature() {
    return temperature;
  },
  get maxTokens() {
    return maxTokens;
  },
  get topK() {
    return topK;
  },
  // Setters
  set temperature(v) {
    temperature = v;
  },
  set maxTokens(v) {
    maxTokens = v;
  },
  set topK(v) {
    topK = v;
  }
};
function ChatInterface($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let inputText = "";
    $$renderer2.push(`<div class="bg-slate-800 rounded-lg p-4 flex flex-col h-full"><div class="flex items-center justify-between mb-4"><h3 class="text-lg font-semibold text-white">Chat</h3> `);
    if (chatStore.messages.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button class="text-sm text-slate-400 hover:text-slate-200 transition-colors">Clear</button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> <div class="grid grid-cols-3 gap-2 mb-4"><div><label class="block text-xs text-slate-400 mb-1">Temp: ${escape_html(chatStore.temperature.toFixed(1))}</label> <input type="range" min="0.1" max="2" step="0.1"${attr("value", chatStore.temperature)} class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"/></div> <div><label class="block text-xs text-slate-400 mb-1">Tokens: ${escape_html(chatStore.maxTokens)}</label> <input type="range" min="10" max="200" step="10"${attr("value", chatStore.maxTokens)} class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"/></div> <div><label class="block text-xs text-slate-400 mb-1">Top-K: ${escape_html(chatStore.topK)}</label> <input type="range" min="1" max="100" step="1"${attr("value", chatStore.topK)} class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"/></div></div> <div class="flex-1 overflow-y-auto space-y-3 min-h-[200px] max-h-[400px] mb-4">`);
    if (chatStore.messages.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="flex items-center justify-center h-full"><p class="text-slate-500 text-sm">`);
      {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`Load the model to chat`);
      }
      $$renderer2.push(`<!--]--></p></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(chatStore.messages);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let message = each_array[$$index];
        $$renderer2.push(`<div${attr_class(`p-3 rounded-lg ${stringify(message.role === "user" ? "bg-blue-900/50 ml-8" : "bg-slate-700 mr-8")}`)}><p class="text-xs text-slate-400 mb-1">${escape_html(message.role === "user" ? "You" : "GPT-2")}</p> <p class="text-slate-200 text-sm whitespace-pre-wrap">${escape_html(message.content)}</p></div>`);
      }
      $$renderer2.push(`<!--]--> `);
      {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <form class="flex gap-2"><input type="text"${attr("value", inputText)}${attr("placeholder", "Load model to chat")}${attr("disabled", true, true)} class="flex-1 bg-slate-700 text-white rounded-md px-3 py-2 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"/> <button type="submit"${attr("disabled", true, true)} class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed">`);
    {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`Send`);
    }
    $$renderer2.push(`<!--]--></button></form></div>`);
  });
}
function ModelStatus($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="bg-slate-800 rounded-lg p-4 mb-6"><div class="flex items-center justify-between"><div class="flex items-center gap-3"><div${attr_class(`w-3 h-3 rounded-full ${stringify("bg-slate-500")}`)}></div> <span class="text-slate-200 font-medium">`);
    {
      $$renderer2.push("<!--[!-->");
      {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`Model Not Loaded`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></span> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> <div class="flex items-center gap-2">`);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"${attr("disabled", modelStore.webgpuSupported === false, true)}>${escape_html("Download Model")}</button>`);
    }
    $$renderer2.push(`<!--]--></div></div> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function DownloadButton($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function _page($$renderer) {
  $$renderer.push(`<div class="container mx-auto p-6 max-w-7xl"><header class="mb-8"><h1 class="text-4xl font-bold text-white mb-2">GPT-2 LoRA Trainer</h1> <p class="text-slate-400">Train a LoRA adapter on your text data, entirely in the browser using WebGPU</p></header> `);
  ModelStatus($$renderer);
  $$renderer.push(`<!----> <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6"><div class="space-y-6">`);
  LoRAConfig($$renderer);
  $$renderer.push(`<!----> `);
  FileDropZone($$renderer);
  $$renderer.push(`<!----> `);
  TrainingProgress($$renderer);
  $$renderer.push(`<!----> `);
  DownloadButton($$renderer);
  $$renderer.push(`<!----></div> <div>`);
  ChatInterface($$renderer);
  $$renderer.push(`<!----></div></div> <footer class="mt-12 text-center text-slate-500 text-sm"><p>Powered by <a href="https://github.com/anthropics/torchlette" class="text-blue-400 hover:underline">Torchlette</a> - WebGPU Tensor Library</p></footer></div>`);
}
export {
  _page as default
};
