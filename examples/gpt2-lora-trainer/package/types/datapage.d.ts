/**
 * Read a data page from uncompressed reader.
 *
 * @param {Uint8Array} bytes raw page data (should already be decompressed)
 * @param {DataPageHeader} daph data page header
 * @param {ColumnDecoder} columnDecoder
 * @returns {DataPage} definition levels, repetition levels, and array of values
 */
export function readDataPage(bytes: Uint8Array, daph: DataPageHeader, { type, element, schemaPath }: ColumnDecoder): DataPage;
/**
 * @param {Uint8Array} compressedBytes
 * @param {number} uncompressed_page_size
 * @param {CompressionCodec} codec
 * @param {Compressors | undefined} compressors
 * @returns {Uint8Array}
 */
export function decompressPage(compressedBytes: Uint8Array, uncompressed_page_size: number, codec: CompressionCodec, compressors: Compressors | undefined): Uint8Array;
/**
 * Read a data page from the given Uint8Array.
 *
 * @param {Uint8Array} compressedBytes raw page data
 * @param {PageHeader} ph page header
 * @param {ColumnDecoder} columnDecoder
 * @returns {DataPage} definition levels, repetition levels, and array of values
 */
export function readDataPageV2(compressedBytes: Uint8Array, ph: PageHeader, columnDecoder: ColumnDecoder): DataPage;
import type { DataPageHeader } from '../src/types.d.ts';
import type { ColumnDecoder } from '../src/types.d.ts';
import type { DataPage } from '../src/types.d.ts';
import type { CompressionCodec } from '../src/types.d.ts';
import type { Compressors } from '../src/types.d.ts';
import type { PageHeader } from '../src/types.d.ts';
//# sourceMappingURL=datapage.d.ts.map