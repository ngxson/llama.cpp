import { config } from '$lib/stores/settings.svelte';

/**
 * PropsService - Server properties management
 *
 * This service handles communication with the /props endpoint to retrieve
 * server configuration, model information, and capabilities.
 *
 * **Responsibilities:**
 * - Fetch server properties from /props endpoint
 * - Handle API authentication
 * - Parse and validate server response
 *
 * **Used by:**
 * - ServerStore: Primary consumer for server state management
 */
export class PropsService {
	/**
	 * Fetches server properties from the /props endpoint
	 *
	 * @returns {Promise<ApiLlamaCppServerProps>} Server properties
	 * @throws {Error} If the request fails or returns invalid data
	 */
	static async fetch(): Promise<ApiLlamaCppServerProps> {
		const currentConfig = config();
		const apiKey = currentConfig.apiKey?.toString().trim();

		const response = await fetch('./props', {
			headers: {
				...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
			}
		});

		if (!response.ok) {
			throw new Error(
				`Failed to fetch server properties: ${response.status} ${response.statusText}`
			);
		}

		const data = await response.json();
		return data as ApiLlamaCppServerProps;
	}
}
