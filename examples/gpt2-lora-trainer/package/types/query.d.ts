/**
 * @import {BaseParquetReadOptions} from '../src/types.js'
 */
/**
 * Wraps parquetRead with orderBy support.
 * This is a parquet-aware query engine that can read a subset of rows and columns.
 * Accepts optional orderBy column name to sort the results.
 * Note that using orderBy may SIGNIFICANTLY increase the query time.
 *
 * @param {BaseParquetReadOptions & { orderBy?: string }} options
 * @returns {Promise<Record<string, any>[]>} resolves when all requested rows and columns are parsed
 */
export function parquetQuery(options: BaseParquetReadOptions & {
    orderBy?: string;
}): Promise<Record<string, any>[]>;
import type { BaseParquetReadOptions } from '../src/types.js';
//# sourceMappingURL=query.d.ts.map