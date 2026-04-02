/**
 * @import {DataReader} from '../src/types.d.ts'
 * @param {DataReader} reader
 * @param {number} count number of values to read
 * @param {Int32Array | BigInt64Array} output
 */
export function deltaBinaryUnpack(reader: DataReader, count: number, output: Int32Array | BigInt64Array): void;
/**
 * @param {DataReader} reader
 * @param {number} count
 * @param {Uint8Array[]} output
 */
export function deltaLengthByteArray(reader: DataReader, count: number, output: Uint8Array[]): void;
/**
 * @param {DataReader} reader
 * @param {number} count
 * @param {Uint8Array[]} output
 */
export function deltaByteArray(reader: DataReader, count: number, output: Uint8Array[]): void;
import type { DataReader } from '../src/types.d.ts';
//# sourceMappingURL=delta.d.ts.map