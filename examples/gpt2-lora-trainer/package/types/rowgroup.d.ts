/**
 * @import {AsyncColumn, AsyncRowGroup, DecodedArray, GroupPlan, ParquetParsers, ParquetReadOptions, QueryPlan, SchemaTree} from './types.js'
 */
/**
 * Read a row group from a file-like object.
 *
 * @param {ParquetReadOptions} options
 * @param {QueryPlan} plan
 * @param {GroupPlan} groupPlan
 * @returns {AsyncRowGroup} resolves to column data
 */
export function readRowGroup(options: ParquetReadOptions, { metadata }: QueryPlan, groupPlan: GroupPlan): AsyncRowGroup;
/**
 * @overload
 * @param {AsyncRowGroup} asyncGroup
 * @param {number} selectStart
 * @param {number} selectEnd
 * @param {string[] | undefined} columns
 * @param {'object'} rowFormat
 * @returns {Promise<Record<string, any>[]>} resolves to row data
 */
export function asyncGroupToRows(asyncGroup: AsyncRowGroup, selectStart: number, selectEnd: number, columns: string[] | undefined, rowFormat: "object"): Promise<Record<string, any>[]>;
/**
 * @overload
 * @param {AsyncRowGroup} asyncGroup
 * @param {number} selectStart
 * @param {number} selectEnd
 * @param {string[] | undefined} columns
 * @param {'array'} [rowFormat]
 * @returns {Promise<any[][]>} resolves to row data
 */
export function asyncGroupToRows(asyncGroup: AsyncRowGroup, selectStart: number, selectEnd: number, columns: string[] | undefined, rowFormat?: "array" | undefined): Promise<any[][]>;
/**
 * Assemble physical columns into top-level columns asynchronously.
 *
 * @param {AsyncRowGroup} asyncRowGroup
 * @param {SchemaTree} schemaTree
 * @param {ParquetParsers} [parsers]
 * @returns {AsyncRowGroup}
 */
export function assembleAsync(asyncRowGroup: AsyncRowGroup, schemaTree: SchemaTree, parsers?: ParquetParsers): AsyncRowGroup;
import type { ParquetReadOptions } from './types.js';
import type { QueryPlan } from './types.js';
import type { GroupPlan } from './types.js';
import type { AsyncRowGroup } from './types.js';
import type { SchemaTree } from './types.js';
import type { ParquetParsers } from './types.js';
//# sourceMappingURL=rowgroup.d.ts.map