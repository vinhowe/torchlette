/**
 * WKB (Well-Known Binary) decoder for geometry objects.
 *
 * @import {DataReader, Geometry} from '../src/types.js'
 * @param {DataReader} reader
 * @returns {Geometry} geometry object
 */
export function wkbToGeojson(reader: DataReader): Geometry;
export type WkbFlags = {
    littleEndian: boolean;
    type: number;
    dim: number;
    count: number;
};
import type { DataReader } from '../src/types.js';
import type { Geometry } from '../src/types.js';
//# sourceMappingURL=wkb.d.ts.map