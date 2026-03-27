/**
 * Read parquet metadata from an async buffer.
 *
 * An AsyncBuffer is like an ArrayBuffer, but the slices are loaded
 * asynchronously, possibly over the network.
 *
 * You must provide the byteLength of the buffer, typically from a HEAD request.
 *
 * In theory, you could use suffix-range requests to fetch the end of the file,
 * and save a round trip. But in practice, this doesn't work because chrome
 * deems suffix-range requests as a not-safe-listed header, and will require
 * a pre-flight. So the byteLength is required.
 *
 * To make this efficient, we initially request the last 512kb of the file,
 * which is likely to contain the metadata. If the metadata length exceeds the
 * initial fetch, 512kb, we request the rest of the metadata from the AsyncBuffer.
 *
 * This ensures that we either make one 512kb initial request for the metadata,
 * or a second request for up to the metadata size.
 *
 * @param {AsyncBuffer} asyncBuffer parquet file contents
 * @param {MetadataOptions & { initialFetchSize?: number }} options initial fetch size in bytes (default 512kb)
 * @returns {Promise<FileMetaData>} parquet metadata object
 */
export function parquetMetadataAsync(asyncBuffer: AsyncBuffer, { parsers, initialFetchSize, geoparquet }?: MetadataOptions & {
    initialFetchSize?: number;
}): Promise<FileMetaData>;
/**
 * Read parquet metadata from a buffer synchronously.
 *
 * @import {KeyValue} from '../src/types.d.ts'
 * @param {ArrayBuffer} arrayBuffer parquet file footer
 * @param {MetadataOptions} options metadata parsing options
 * @returns {FileMetaData} parquet metadata object
 */
export function parquetMetadata(arrayBuffer: ArrayBuffer, { parsers, geoparquet }?: MetadataOptions): FileMetaData;
/**
 * Return a tree of schema elements from parquet metadata.
 *
 * @param {{schema: SchemaElement[]}} metadata parquet metadata object
 * @returns {SchemaTree} tree of schema elements
 */
export function parquetSchema({ schema }: {
    schema: SchemaElement[];
}): SchemaTree;
/**
 * @param {Uint8Array | undefined} value
 * @param {SchemaElement} schema
 * @param {ParquetParsers} parsers
 * @returns {MinMaxType | undefined}
 */
export function convertMetadata(value: Uint8Array | undefined, schema: SchemaElement, parsers: ParquetParsers): MinMaxType | undefined;
export const defaultInitialFetchSize: number;
import type { AsyncBuffer } from '../src/types.d.ts';
import type { MetadataOptions } from '../src/types.d.ts';
import type { FileMetaData } from '../src/types.d.ts';
import type { SchemaElement } from '../src/types.d.ts';
import type { SchemaTree } from '../src/types.d.ts';
import type { ParquetParsers } from '../src/types.d.ts';
import type { MinMaxType } from '../src/types.d.ts';
//# sourceMappingURL=metadata.d.ts.map