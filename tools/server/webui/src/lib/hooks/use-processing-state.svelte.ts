import {
	subscribeToProcessingState,
	getCurrentProcessingState,
	isChatStreaming
} from '$lib/stores/chat.svelte';
import { config } from '$lib/stores/settings.svelte';

export interface UseProcessingStateReturn {
	readonly processingState: ApiProcessingState | null;
	getProcessingDetails(): string[];
	getProcessingMessage(): string;
	shouldShowDetails(): boolean;
	startMonitoring(): Promise<void>;
	stopMonitoring(): void;
}

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It subscribes to timing data updates from ChatStore and provides
 * formatted processing details for UI display.
 *
 * **Features:**
 * - Real-time processing state monitoring
 * - Context and output token tracking
 * - Tokens per second calculation
 * - Graceful degradation when slots endpoint unavailable
 * - Automatic cleanup on component unmount
 *
 * @returns Hook interface with processing state and control methods
 */
export function useProcessingState(): UseProcessingStateReturn {
	let isMonitoring = $state(false);
	let processingState = $state<ApiProcessingState | null>(null);
	let lastKnownState = $state<ApiProcessingState | null>(null);
	let unsubscribe: (() => void) | null = null;

	async function startMonitoring(): Promise<void> {
		if (isMonitoring) return;

		isMonitoring = true;

		unsubscribe = subscribeToProcessingState((state) => {
			processingState = state;
			if (state) {
				lastKnownState = state;
			} else {
				lastKnownState = null;
			}
		});

		try {
			const currentState = await getCurrentProcessingState();

			if (currentState) {
				processingState = currentState;
				lastKnownState = currentState;
			}

			// Check if streaming is active for UI purposes
			if (isChatStreaming()) {
				// Streaming is active, state will be updated via subscription
			}
		} catch (error) {
			console.warn('Failed to start processing state monitoring:', error);
			// Continue without monitoring - graceful degradation
		}
	}

	function stopMonitoring(): void {
		if (!isMonitoring) return;

		isMonitoring = false;

		// Only clear processing state if keepStatsVisible is disabled
		// This preserves the last known state for display when stats should remain visible
		const currentConfig = config();
		if (!currentConfig.keepStatsVisible) {
			processingState = null;
		} else if (lastKnownState) {
			// Keep the last known state visible when keepStatsVisible is enabled
			processingState = lastKnownState;
		}

		if (unsubscribe) {
			unsubscribe();
			unsubscribe = null;
		}
	}

	function getProcessingMessage(): string {
		if (!processingState) {
			return 'Processing...';
		}

		switch (processingState.status) {
			case 'initializing':
				return 'Initializing...';
			case 'preparing':
				if (processingState.progressPercent !== undefined) {
					return `Processing (${processingState.progressPercent}%)`;
				}
				return 'Preparing response...';
			case 'generating':
				if (processingState.tokensDecoded > 0) {
					return `Generating... (${processingState.tokensDecoded} tokens)`;
				}
				return 'Generating...';
			default:
				return 'Processing...';
		}
	}

	function getProcessingDetails(): string[] {
		// Use current processing state or fall back to last known state
		const stateToUse = processingState || lastKnownState;
		if (!stateToUse) {
			return [];
		}

		const details: string[] = [];

		// Always show context info when we have valid data
		if (stateToUse.contextUsed >= 0 && stateToUse.contextTotal > 0) {
			const contextPercent = Math.round((stateToUse.contextUsed / stateToUse.contextTotal) * 100);

			details.push(
				`Context: ${stateToUse.contextUsed}/${stateToUse.contextTotal} (${contextPercent}%)`
			);
		}

		if (stateToUse.outputTokensUsed > 0) {
			// Handle infinite max_tokens (-1) case
			if (stateToUse.outputTokensMax <= 0) {
				details.push(`Output: ${stateToUse.outputTokensUsed}/∞`);
			} else {
				const outputPercent = Math.round(
					(stateToUse.outputTokensUsed / stateToUse.outputTokensMax) * 100
				);

				details.push(
					`Output: ${stateToUse.outputTokensUsed}/${stateToUse.outputTokensMax} (${outputPercent}%)`
				);
			}
		}

		if (stateToUse.tokensPerSecond && stateToUse.tokensPerSecond > 0) {
			details.push(`${stateToUse.tokensPerSecond.toFixed(1)} tokens/sec`);
		}

		if (stateToUse.speculative) {
			details.push('Speculative decoding enabled');
		}

		return details;
	}

	function shouldShowDetails(): boolean {
		return processingState !== null && processingState.status !== 'idle';
	}

	return {
		get processingState() {
			return processingState;
		},
		getProcessingDetails,
		getProcessingMessage,
		shouldShowDetails,
		startMonitoring,
		stopMonitoring
	};
}
