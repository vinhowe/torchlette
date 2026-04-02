/**
 * Reconstructs a complex nested structure from flat arrays of values and
 * definition and repetition levels, according to Dremel encoding.
 *
 * @param {any[]} output
 * @param {number[] | undefined} definitionLevels
 * @param {number[]} repetitionLevels
 * @param {DecodedArray} values
 * @param {SchemaTree[]} schemaPath
 * @returns {DecodedArray}
 */
export function assembleLists(output: any[], definitionLevels: number[] | undefined, repetitionLevels: number[], values: DecodedArray, schemaPath: SchemaTree[]): DecodedArray;
/**
 * Assemble a nested structure from subcolumn data.
 *
 * @param {Map<string, DecodedArray>} subcolumnData
 * @param {SchemaTree} schema top-level schema element
 * @param {ParquetParsers} parsers
 * @param {number} [depth] depth of nested structure
 */
export function assembleNested(subcolumnData: Map<string, DecodedArray>, schema: SchemaTree, parsers: ParquetParsers, depth?: number): void;
import type { DecodedArray } from '../src/types.d.ts';
import type { SchemaTree } from '../src/types.d.ts';
import type { ParquetParsers } from '../src/types.d.ts';
//# sourceMappingURL=assemble.d.ts.map