/**
 * IndexedDB cache for model weights and tokenizer.
 *
 * Stores large binary data locally to avoid re-downloading on page reload.
 */

const DB_NAME = 'gpt2-lora-cache';
const DB_VERSION = 1;
const WEIGHTS_STORE = 'weights';
const TOKENIZER_STORE = 'tokenizer';

let db: IDBDatabase | null = null;

/**
 * Open the IndexedDB database.
 */
async function openDB(): Promise<IDBDatabase> {
  if (db) return db;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);

    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      const database = (event.target as IDBOpenDBRequest).result;

      // Create stores if they don't exist
      if (!database.objectStoreNames.contains(WEIGHTS_STORE)) {
        database.createObjectStore(WEIGHTS_STORE);
      }
      if (!database.objectStoreNames.contains(TOKENIZER_STORE)) {
        database.createObjectStore(TOKENIZER_STORE);
      }
    };
  });
}

/**
 * Store model weights in cache.
 */
export async function cacheWeights(buffer: ArrayBuffer): Promise<void> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(WEIGHTS_STORE, 'readwrite');
    const store = transaction.objectStore(WEIGHTS_STORE);

    // Store the raw buffer with a timestamp
    const request = store.put(
      {
        buffer,
        timestamp: Date.now(),
        version: 1,
      },
      'gpt2-weights'
    );

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

/**
 * Get cached model weights.
 */
export async function getCachedWeights(): Promise<ArrayBuffer | null> {
  try {
    const database = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(WEIGHTS_STORE, 'readonly');
      const store = transaction.objectStore(WEIGHTS_STORE);
      const request = store.get('gpt2-weights');

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const result = request.result;
        if (result && result.buffer) {
          resolve(result.buffer);
        } else {
          resolve(null);
        }
      };
    });
  } catch {
    return null;
  }
}

/**
 * Store tokenizer data in cache.
 */
export async function cacheTokenizer(data: {
  vocab: Record<string, number>;
  merges: string[];
}): Promise<void> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(TOKENIZER_STORE, 'readwrite');
    const store = transaction.objectStore(TOKENIZER_STORE);

    const request = store.put(
      {
        ...data,
        timestamp: Date.now(),
        version: 1,
      },
      'gpt2-tokenizer'
    );

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

/**
 * Get cached tokenizer data.
 */
export async function getCachedTokenizer(): Promise<{
  vocab: Record<string, number>;
  merges: string[];
} | null> {
  try {
    const database = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = database.transaction(TOKENIZER_STORE, 'readonly');
      const store = transaction.objectStore(TOKENIZER_STORE);
      const request = store.get('gpt2-tokenizer');

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const result = request.result;
        if (result && result.vocab && result.merges) {
          resolve({ vocab: result.vocab, merges: result.merges });
        } else {
          resolve(null);
        }
      };
    });
  } catch {
    return null;
  }
}

/**
 * Clear all cached data.
 */
export async function clearCache(): Promise<void> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(
      [WEIGHTS_STORE, TOKENIZER_STORE],
      'readwrite'
    );

    transaction.objectStore(WEIGHTS_STORE).clear();
    transaction.objectStore(TOKENIZER_STORE).clear();

    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

/**
 * Check if weights are cached.
 */
export async function hasWeightsCache(): Promise<boolean> {
  const weights = await getCachedWeights();
  return weights !== null;
}

/**
 * Check if tokenizer is cached.
 */
export async function hasTokenizerCache(): Promise<boolean> {
  const tokenizer = await getCachedTokenizer();
  return tokenizer !== null;
}

/**
 * Get cache size info.
 */
export async function getCacheInfo(): Promise<{
  hasWeights: boolean;
  hasTokenizer: boolean;
  weightsSize: number;
}> {
  const weights = await getCachedWeights();
  const tokenizer = await getCachedTokenizer();

  return {
    hasWeights: weights !== null,
    hasTokenizer: tokenizer !== null,
    weightsSize: weights?.byteLength ?? 0,
  };
}
