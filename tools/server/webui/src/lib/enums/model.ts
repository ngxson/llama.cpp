/**
 * Model status enum - matches tools/server/server-models.h from C++ server
 */
export enum ServerModelStatus {
	UNLOADED = 'UNLOADED',
	LOADING = 'LOADING',
	LOADED = 'LOADED',
	FAILED = 'FAILED'
}
