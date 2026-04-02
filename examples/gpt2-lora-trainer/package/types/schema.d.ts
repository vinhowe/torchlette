/**
 * Get schema elements from the root to the given element name.
 *
 * @param {SchemaElement[]} schema
 * @param {string[]} name path to the element
 * @returns {SchemaTree[]} list of schema elements
 */
export function getSchemaPath(schema: SchemaElement[], name: string[]): SchemaTree[];
/**
 * Get all physical (leaf) column names.
 *
 * @param {SchemaTree} schemaTree
 * @returns {string[]} list of physical column names
 */
export function getPhysicalColumns(schemaTree: SchemaTree): string[];
/**
 * Get the max repetition level for a given schema path.
 *
 * @param {SchemaTree[]} schemaPath
 * @returns {number} max repetition level
 */
export function getMaxRepetitionLevel(schemaPath: SchemaTree[]): number;
/**
 * Get the max definition level for a given schema path.
 *
 * @param {SchemaTree[]} schemaPath
 * @returns {number} max definition level
 */
export function getMaxDefinitionLevel(schemaPath: SchemaTree[]): number;
/**
 * Check if a column is list-like.
 *
 * @param {SchemaTree} schema
 * @returns {boolean} true if list-like
 */
export function isListLike(schema: SchemaTree): boolean;
/**
 * Check if a column is map-like.
 *
 * @param {SchemaTree} schema
 * @returns {boolean} true if map-like
 */
export function isMapLike(schema: SchemaTree): boolean;
/**
 * Returns true if a column is non-nested.
 *
 * @param {SchemaTree[]} schemaPath
 * @returns {boolean}
 */
export function isFlatColumn(schemaPath: SchemaTree[]): boolean;
import type { SchemaElement } from '../src/types.d.ts';
import type { SchemaTree } from '../src/types.d.ts';
//# sourceMappingURL=schema.d.ts.map