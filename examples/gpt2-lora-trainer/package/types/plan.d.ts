/**
 * @import {AsyncBuffer, ByteRange, ChunkPlan, GroupPlan, ParquetReadOptions, QueryPlan} from '../src/types.js'
 */
/**
 * Plan which byte ranges to read to satisfy a read request.
 * Metadata must be non-null.
 *
 * @param {ParquetReadOptions} options
 * @returns {QueryPlan}
 */
export function parquetPlan({ metadata, rowStart, rowEnd, columns, filter, filterStrict, useOffsetIndex }: ParquetReadOptions): QueryPlan;
/**
 * Prefetch byte ranges from an AsyncBuffer.
 *
 * @param {AsyncBuffer} file
 * @param {QueryPlan} plan
 * @returns {AsyncBuffer}
 */
export function prefetchAsyncBuffer(file: AsyncBuffer, { fetches }: QueryPlan): AsyncBuffer;
import type { ParquetReadOptions } from '../src/types.js';
import type { QueryPlan } from '../src/types.js';
import type { AsyncBuffer } from '../src/types.js';
//# sourceMappingURL=plan.d.ts.map