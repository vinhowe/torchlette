/**
 * Read `count` values of the given type from the reader.view.
 *
 * @param {DataReader} reader - buffer to read data from
 * @param {ParquetType} type - parquet type of the data
 * @param {number} count - number of values to read
 * @param {number | undefined} fixedLength - length of each fixed length byte array
 * @returns {DecodedArray} array of values
 */
export function readPlain(reader: DataReader, type: ParquetType, count: number, fixedLength: number | undefined): DecodedArray;
import type { DataReader } from '../src/types.d.ts';
import type { ParquetType } from '../src/types.d.ts';
import type { DecodedArray } from '../src/types.d.ts';
//# sourceMappingURL=plain.d.ts.map