/**
 * Parse column data from a buffer.
 *
 * @param {DataReader} reader
 * @param {RowGroupSelect} rowGroupSelect row group selection
 * @param {ColumnDecoder} columnDecoder column decoder params
 * @param {(chunk: SubColumnData) => void} [onPage] callback for each page
 * @returns {{ data: DecodedArray[], skipped: number }}
 */
export function readColumn(reader: DataReader, { groupStart, selectStart, selectEnd }: RowGroupSelect, columnDecoder: ColumnDecoder, onPage?: (chunk: SubColumnData) => void): {
    data: DecodedArray[];
    skipped: number;
};
/**
 * Read a page (data or dictionary) from a buffer.
 *
 * @import {PageResult} from '../src/types.d.ts'
 * @param {DataReader} reader
 * @param {PageHeader} header
 * @param {ColumnDecoder} columnDecoder
 * @param {DecodedArray | undefined} dictionary
 * @param {DecodedArray | undefined} previousChunk
 * @param {number} pageStart skip this many rows in the page
 * @returns {PageResult}
 */
export function readPage(reader: DataReader, header: PageHeader, columnDecoder: ColumnDecoder, dictionary: DecodedArray | undefined, previousChunk: DecodedArray | undefined, pageStart: number): PageResult;
import type { DataReader } from '../src/types.d.ts';
import type { RowGroupSelect } from '../src/types.d.ts';
import type { ColumnDecoder } from '../src/types.d.ts';
import type { SubColumnData } from '../src/types.d.ts';
import type { DecodedArray } from '../src/types.d.ts';
import type { PageHeader } from '../src/types.d.ts';
import type { PageResult } from '../src/types.d.ts';
//# sourceMappingURL=column.d.ts.map