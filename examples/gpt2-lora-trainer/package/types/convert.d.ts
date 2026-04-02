/**
 * Convert known types from primitive to rich, and dereference dictionary.
 *
 * @param {DecodedArray} data series of primitive types
 * @param {DecodedArray | undefined} dictionary
 * @param {Encoding} encoding
 * @param {ColumnDecoder} columnDecoder
 * @returns {DecodedArray} series of rich types
 */
export function convertWithDictionary(data: DecodedArray, dictionary: DecodedArray | undefined, encoding: Encoding, columnDecoder: ColumnDecoder): DecodedArray;
/**
 * Convert known types from primitive to rich.
 *
 * @param {DecodedArray} data series of primitive types
 * @param {ColumnDecoder} columnDecoder
 * @returns {DecodedArray} series of rich types
 */
export function convert(data: DecodedArray, columnDecoder: ColumnDecoder): DecodedArray;
/**
 * @param {Uint8Array} bytes
 * @returns {number}
 */
export function parseDecimal(bytes: Uint8Array): number;
/**
 * @param {Uint8Array | undefined} bytes
 * @returns {number | undefined}
 */
export function parseFloat16(bytes: Uint8Array | undefined): number | undefined;
/**
 * Default type parsers when no custom ones are given
 * @type ParquetParsers
 */
export const DEFAULT_PARSERS: ParquetParsers;
import type { DecodedArray } from '../src/types.js';
import type { Encoding } from '../src/types.js';
import type { ColumnDecoder } from '../src/types.js';
import type { ParquetParsers } from '../src/types.js';
//# sourceMappingURL=convert.d.ts.map