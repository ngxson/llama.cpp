import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database';
import type {
	DatabaseConversation,
	DatabaseMessage,
	ExportedConversations
} from '$lib/types/database';

/**
 * ConversationsService - Database operations for persistent conversation management
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. Represents the
 *   real-time streaming session and UI visualization. Managed by ChatService/Store.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   This service handles all database operations for conversations - they survive across
 *   sessions, page reloads, and browser restarts. Contains message history, branching
 *   structure, timestamps, and conversation metadata.
 *
 * This service handles all conversation-level database operations including CRUD,
 * import/export, and navigation. It provides a stateless abstraction layer between
 * ConversationsStore and DatabaseService.
 *
 * **Architecture & Relationships:**
 * - **ConversationsService** (this class): Stateless database operations layer
 *   - Handles conversation CRUD operations via DatabaseService
 *   - Manages import/export with JSON serialization
 *   - Provides navigation helpers for routing
 *   - Does not manage reactive UI state
 *
 * - **ConversationsStore**: Uses this service for all database operations
 * - **DatabaseService**: Low-level IndexedDB operations
 * - **ChatStore**: Indirectly uses conversation data for AI context
 *
 * **Key Responsibilities:**
 * - Conversation CRUD (create, read, update, delete)
 * - Message retrieval for conversations
 * - Import/export with file download/upload
 * - Navigation helpers (goto conversation, new chat)
 */
export class ConversationsService {
	/**
	 * Creates a new conversation in the database
	 * @param name - Optional name for the conversation, defaults to timestamped name
	 * @returns The created conversation
	 */
	async createConversation(name?: string): Promise<DatabaseConversation> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		return await DatabaseService.createConversation(conversationName);
	}

	/**
	 * Loads all conversations from the database
	 * @returns Array of all conversations
	 */
	async loadAllConversations(): Promise<DatabaseConversation[]> {
		return await DatabaseService.getAllConversations();
	}

	/**
	 * Loads a specific conversation by ID
	 * @param convId - The conversation ID to load
	 * @returns The conversation or null if not found
	 */
	async loadConversation(convId: string): Promise<DatabaseConversation | null> {
		return await DatabaseService.getConversation(convId);
	}

	/**
	 * Gets all messages for a conversation
	 * @param convId - The conversation ID
	 * @returns Array of messages in the conversation
	 */
	async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await DatabaseService.getConversationMessages(convId);
	}

	/**
	 * Updates the name of a conversation
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		await DatabaseService.updateConversation(convId, { name });
	}

	/**
	 * Updates the current node (currNode) of a conversation
	 * @param convId - The conversation ID to update
	 * @param nodeId - The new current node ID
	 */
	async updateCurrentNode(convId: string, nodeId: string): Promise<void> {
		await DatabaseService.updateCurrentNode(convId, nodeId);
	}

	/**
	 * Updates the lastModified timestamp of a conversation
	 * @param convId - The conversation ID to update
	 */
	async updateTimestamp(convId: string): Promise<void> {
		await DatabaseService.updateConversation(convId, { lastModified: Date.now() });
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		await DatabaseService.deleteConversation(convId);
	}

	/**
	 * Downloads a conversation as JSON file
	 * @param conversation - The conversation to download
	 * @param messages - The messages in the conversation
	 */
	downloadConversation(conversation: DatabaseConversation, messages: DatabaseMessage[]): void {
		const conversationData: ExportedConversations = {
			conv: conversation,
			messages
		};

		this.triggerDownload(conversationData);
	}

	/**
	 * Exports all conversations with their messages as a JSON file
	 * @returns The list of exported conversations
	 */
	async exportAllConversations(): Promise<DatabaseConversation[]> {
		const allConversations = await DatabaseService.getAllConversations();

		if (allConversations.length === 0) {
			throw new Error('No conversations to export');
		}

		const allData: ExportedConversations = await Promise.all(
			allConversations.map(async (conv) => {
				const messages = await DatabaseService.getConversationMessages(conv.id);
				return { conv, messages };
			})
		);

		const blob = new Blob([JSON.stringify(allData, null, 2)], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `all_conversations_${new Date().toISOString().split('T')[0]}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);

		toast.success(`All conversations (${allConversations.length}) prepared for download`);

		return allConversations;
	}

	/**
	 * Imports conversations from a JSON file
	 * Opens file picker and processes the selected file
	 * @returns Promise resolving to the list of imported conversations
	 */
	async importConversations(): Promise<DatabaseConversation[]> {
		return new Promise((resolve, reject) => {
			const input = document.createElement('input');
			input.type = 'file';
			input.accept = '.json';

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];

				if (!file) {
					reject(new Error('No file selected'));
					return;
				}

				try {
					const text = await file.text();
					const parsedData = JSON.parse(text);
					let importedData: ExportedConversations;

					if (Array.isArray(parsedData)) {
						importedData = parsedData;
					} else if (
						parsedData &&
						typeof parsedData === 'object' &&
						'conv' in parsedData &&
						'messages' in parsedData
					) {
						// Single conversation object
						importedData = [parsedData];
					} else {
						throw new Error(
							'Invalid file format: expected array of conversations or single conversation object'
						);
					}

					const result = await DatabaseService.importConversations(importedData);

					toast.success(`Imported ${result.imported} conversation(s), skipped ${result.skipped}`);

					// Extract the conversation objects from imported data
					const importedConversations = (
						Array.isArray(importedData) ? importedData : [importedData]
					).map((item) => item.conv);

					resolve(importedConversations);
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';
					console.error('Failed to import conversations:', err);
					toast.error('Import failed', {
						description: message
					});
					reject(new Error(`Import failed: ${message}`));
				}
			};

			input.click();
		});
	}

	/**
	 * Navigates to a specific conversation route
	 * @param convId - The conversation ID to navigate to
	 */
	async navigateToConversation(convId: string): Promise<void> {
		await goto(`#/chat/${convId}`);
	}

	/**
	 * Navigates to new chat route
	 */
	async navigateToNewChat(): Promise<void> {
		await goto(`?new_chat=true#/`);
	}

	/**
	 * Triggers file download in browser
	 * @param data - Data to download
	 * @param filename - Optional filename
	 */
	private triggerDownload(data: ExportedConversations, filename?: string): void {
		const conversation =
			'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;

		if (!conversation) {
			console.error('Invalid data: missing conversation');
			return;
		}

		const conversationName = conversation.name ? conversation.name.trim() : '';
		const convId = conversation.id || 'unknown';
		const truncatedSuffix = conversationName
			.toLowerCase()
			.replace(/[^a-z0-9]/gi, '_')
			.replace(/_+/g, '_')
			.substring(0, 20);
		const downloadFilename = filename || `conversation_${convId}_${truncatedSuffix}.json`;

		const conversationJson = JSON.stringify(data, null, 2);
		const blob = new Blob([conversationJson], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = downloadFilename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}
}

export const conversationsService = new ConversationsService();
