/**
 * @import {ParquetQueryFilter, RowGroup} from '../src/types.js'
 */
/**
 * Returns an array of top-level column names needed to evaluate the filter.
 *
 * @param {ParquetQueryFilter} [filter]
 * @returns {string[]}
 */
export function columnsNeededForFilter(filter?: ParquetQueryFilter): string[];
/**
 * Match a record against a query filter
 *
 * @param {Record<string, any>} record
 * @param {ParquetQueryFilter} filter
 * @param {boolean} [strict]
 * @returns {boolean}
 */
export function matchFilter(record: Record<string, any>, filter: ParquetQueryFilter, strict?: boolean): boolean;
/**
 * Check if a row group can be skipped based on filter and column statistics.
 *
 * @param {object} options
 * @param {RowGroup} options.rowGroup
 * @param {string[]} options.physicalColumns
 * @param {ParquetQueryFilter | undefined} options.filter
 * @param {boolean} [options.strict]
 * @returns {boolean} true if the row group can be skipped
 */
export function canSkipRowGroup({ rowGroup, physicalColumns, filter, strict }: {
    rowGroup: RowGroup;
    physicalColumns: string[];
    filter: ParquetQueryFilter | undefined;
    strict?: boolean | undefined;
}): boolean;
import type { ParquetQueryFilter } from '../src/types.js';
import type { RowGroup } from '../src/types.js';
//# sourceMappingURL=filter.d.ts.map