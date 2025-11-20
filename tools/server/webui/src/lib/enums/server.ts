/**
 * Server mode enum - used for single/multi-model mode
 */
export enum ServerMode {
	/** Single model mode - server running with a specific model loaded */
	MODEL = 'MODEL',
	/** Router mode - server managing multiple model instances */
	ROUTER = 'ROUTER'
}
