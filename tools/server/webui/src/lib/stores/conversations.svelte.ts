import { browser } from '$app/environment';
import { conversationsService } from '$lib/services/conversations';
import { slotsService } from '$lib/services/slots';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode } from '$lib/utils/branching';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

/**
 * ConversationsStore - Persistent conversation data and lifecycle management
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. Represents the
 *   real-time streaming session, loading states, and UI visualization of AI communication.
 *   Managed by ChatStore, a "chat" is ephemeral and exists during active AI interactions.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   A "conversation" survives across sessions, page reloads, and browser restarts.
 *   It contains the complete message history, branching structure, and conversation metadata.
 *
 * This store manages all conversation-level data and operations including creation, loading,
 * deletion, and navigation. It maintains the list of conversations and the currently active
 * conversation with its message history, providing reactive state for UI components.
 *
 * **Architecture & Relationships:**
 * - **ConversationsStore** (this class): Persistent conversation data management
 *   - Manages conversation list and active conversation state
 *   - Handles conversation CRUD operations via ConversationsService
 *   - Maintains active message array for current conversation
 *   - Coordinates branching navigation (currNode tracking)
 *
 * - **ChatStore**: Uses conversation data as context for active AI streaming
 * - **ConversationsService**: Database operations for conversation persistence
 * - **SlotsService**: Notified of active conversation changes
 * - **DatabaseService**: Low-level storage for conversations and messages
 *
 * **Key Features:**
 * - **Conversation Lifecycle**: Create, load, update, delete conversations
 * - **Message Management**: Active message array with branching support
 * - **Import/Export**: JSON-based conversation backup and restore
 * - **Branch Navigation**: Navigate between message tree branches
 * - **Title Management**: Auto-update titles with confirmation dialogs
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 *
 * **State Properties:**
 * - `conversations`: All conversations sorted by last modified
 * - `activeConversation`: Currently viewed conversation
 * - `activeMessages`: Messages in current conversation path
 * - `isInitialized`: Store initialization status
 */
class ConversationsStore {
	/** List of all conversations */
	conversations = $state<DatabaseConversation[]>([]);

	/** Currently active conversation */
	activeConversation = $state<DatabaseConversation | null>(null);

	/** Messages in the active conversation (filtered by currNode path) */
	activeMessages = $state<DatabaseMessage[]>([]);

	/** Whether the store has been initialized */
	isInitialized = $state(false);

	/** Callback for title update confirmation dialog */
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	/**
	 * Initializes the conversations store by loading conversations from the database
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadConversations();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize conversations store:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 */
	async loadConversations(): Promise<void> {
		this.conversations = await conversationsService.loadAllConversations();
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversation = await conversationsService.createConversation(name);

		this.conversations.unshift(conversation);
		this.activeConversation = conversation;
		this.activeMessages = [];

		slotsService.setActiveConversation(conversation.id);

		await conversationsService.navigateToConversation(conversation.id);

		return conversation.id;
	}

	/**
	 * Loads a specific conversation and its messages
	 * @param convId - The conversation ID to load
	 * @returns True if conversation was loaded successfully
	 */
	async loadConversation(convId: string): Promise<boolean> {
		try {
			const conversation = await conversationsService.loadConversation(convId);

			if (!conversation) {
				return false;
			}

			this.activeConversation = conversation;

			slotsService.setActiveConversation(convId);

			if (conversation.currNode) {
				const allMessages = await conversationsService.getConversationMessages(convId);
				this.activeMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
			} else {
				// Load all messages for conversations without currNode (backward compatibility)
				this.activeMessages = await conversationsService.getConversationMessages(convId);
			}

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);
			return false;
		}
	}

	/**
	 * Clears the active conversation and messages
	 * Used when navigating away from chat or starting fresh
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
		slotsService.setActiveConversation(null);
	}

	/**
	 * Refreshes active messages based on currNode after branch navigation
	 */
	async refreshActiveMessages(): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await conversationsService.getConversationMessages(
			this.activeConversation.id
		);

		if (allMessages.length === 0) {
			this.activeMessages = [];
			return;
		}

		const leafNodeId =
			this.activeConversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];

		this.activeMessages.length = 0;
		this.activeMessages.push(...currentPath);
	}

	/**
	 * Updates the name of a conversation
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await conversationsService.updateConversationName(convId, name);

			const convIndex = this.conversations.findIndex((c) => c.id === convId);

			if (convIndex !== -1) {
				this.conversations[convIndex].name = name;
			}

			if (this.activeConversation?.id === convId) {
				this.activeConversation.name = name;
			}
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	/**
	 * Sets the callback function for title update confirmations
	 * @param callback - Function to call when confirmation is needed
	 */
	setTitleUpdateConfirmationCallback(
		callback: (currentTitle: string, newTitle: string) => Promise<boolean>
	): void {
		this.titleUpdateConfirmationCallback = callback;
	}

	/**
	 * Updates conversation title with optional confirmation dialog based on settings
	 * @param convId - The conversation ID to update
	 * @param newTitle - The new title content
	 * @param onConfirmationNeeded - Callback when user confirmation is needed
	 * @returns True if title was updated, false if cancelled
	 */
	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string,
		onConfirmationNeeded?: (currentTitle: string, newTitle: string) => Promise<boolean>
	): Promise<boolean> {
		try {
			const currentConfig = config();

			if (currentConfig.askForTitleConfirmation && onConfirmationNeeded) {
				const conversation = await conversationsService.loadConversation(convId);
				if (!conversation) return false;

				const shouldUpdate = await onConfirmationNeeded(conversation.name, newTitle);
				if (!shouldUpdate) return false;
			}

			await this.updateConversationName(convId, newTitle);
			return true;
		} catch (error) {
			console.error('Failed to update conversation title with confirmation:', error);
			return false;
		}
	}

	/**
	 * Updates the current node of the active conversation
	 * @param nodeId - The new current node ID
	 */
	async updateCurrentNode(nodeId: string): Promise<void> {
		if (!this.activeConversation) return;

		await conversationsService.updateCurrentNode(this.activeConversation.id, nodeId);
		this.activeConversation.currNode = nodeId;
	}

	/**
	 * Updates conversation lastModified timestamp and moves it to top of list
	 */
	updateConversationTimestamp(): void {
		if (!this.activeConversation) return;

		const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);

		if (chatIndex !== -1) {
			this.conversations[chatIndex].lastModified = Date.now();
			const updatedConv = this.conversations.splice(chatIndex, 1)[0];
			this.conversations.unshift(updatedConv);
		}
	}

	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		// Get the current first user message before navigation
		const allMessages = await conversationsService.getConversationMessages(
			this.activeConversation.id
		);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const currentFirstUserMessage = this.activeMessages.find(
			(m) => m.role === 'user' && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await conversationsService.updateCurrentNode(this.activeConversation.id, currentLeafNodeId);
		this.activeConversation.currNode = currentLeafNodeId;
		await this.refreshActiveMessages();

		// Only show title dialog if we're navigating between different first user message siblings
		if (rootMessage && this.activeMessages.length > 0) {
			const newFirstUserMessage = this.activeMessages.find(
				(m) => m.role === 'user' && m.parent === rootMessage.id
			);

			if (
				newFirstUserMessage &&
				newFirstUserMessage.content.trim() &&
				(!currentFirstUserMessage ||
					newFirstUserMessage.id !== currentFirstUserMessage.id ||
					newFirstUserMessage.content.trim() !== currentFirstUserMessage.content.trim())
			) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newFirstUserMessage.content.trim(),
					this.titleUpdateConfirmationCallback
				);
			}
		}
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		try {
			await conversationsService.deleteConversation(convId);

			this.conversations = this.conversations.filter((c) => c.id !== convId);

			if (this.activeConversation?.id === convId) {
				this.activeConversation = null;
				this.activeMessages = [];
				await conversationsService.navigateToNewChat();
			}
		} catch (error) {
			console.error('Failed to delete conversation:', error);
		}
	}

	/**
	 * Downloads a conversation as JSON file
	 * @param convId - The conversation ID to download
	 */
	async downloadConversation(convId: string): Promise<void> {
		if (this.activeConversation?.id === convId) {
			// Use current active conversation data
			conversationsService.downloadConversation(this.activeConversation, this.activeMessages);
		} else {
			// Load the conversation if not currently active
			const conversation = await conversationsService.loadConversation(convId);
			if (!conversation) return;

			const messages = await conversationsService.getConversationMessages(convId);
			conversationsService.downloadConversation(conversation, messages);
		}
	}

	/**
	 * Exports all conversations with their messages as a JSON file
	 * @returns The list of exported conversations
	 */
	async exportAllConversations(): Promise<DatabaseConversation[]> {
		return await conversationsService.exportAllConversations();
	}

	/**
	 * Imports conversations from a JSON file
	 * @returns The list of imported conversations
	 */
	async importConversations(): Promise<DatabaseConversation[]> {
		const importedConversations = await conversationsService.importConversations();

		// Refresh conversations list after import
		await this.loadConversations();

		return importedConversations;
	}

	/**
	 * Gets all messages for a specific conversation
	 * @param convId - The conversation ID
	 * @returns Array of messages
	 */
	async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await conversationsService.getConversationMessages(convId);
	}

	/**
	 * Adds a message to the active messages array
	 * Used by ChatStore when creating new messages
	 * @param message - The message to add
	 */
	addMessageToActive(message: DatabaseMessage): void {
		this.activeMessages.push(message);
	}

	/**
	 * Updates a message at a specific index in active messages
	 * Creates a new object to trigger Svelte 5 reactivity
	 * @param index - The index of the message to update
	 * @param updates - Partial message data to update
	 */
	updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		if (index !== -1 && this.activeMessages[index]) {
			// Create new object to trigger Svelte 5 reactivity
			this.activeMessages[index] = { ...this.activeMessages[index], ...updates };
		}
	}

	/**
	 * Finds the index of a message in active messages
	 * @param messageId - The message ID to find
	 * @returns The index of the message, or -1 if not found
	 */
	findMessageIndex(messageId: string): number {
		return this.activeMessages.findIndex((m) => m.id === messageId);
	}

	/**
	 * Removes messages from active messages starting at an index
	 * @param startIndex - The index to start removing from
	 */
	sliceActiveMessages(startIndex: number): void {
		this.activeMessages = this.activeMessages.slice(0, startIndex);
	}

	/**
	 * Removes a message from active messages by index
	 * @param index - The index to remove
	 * @returns The removed message or undefined
	 */
	removeMessageAtIndex(index: number): DatabaseMessage | undefined {
		if (index !== -1) {
			return this.activeMessages.splice(index, 1)[0];
		}
		return undefined;
	}
}

export const conversationsStore = new ConversationsStore();

// Export getter functions for reactive access
export const conversations = () => conversationsStore.conversations;
export const activeConversation = () => conversationsStore.activeConversation;
export const activeMessages = () => conversationsStore.activeMessages;
export const isConversationsInitialized = () => conversationsStore.isInitialized;

// Export conversation operations
export const createConversation = conversationsStore.createConversation.bind(conversationsStore);
export const loadConversation = conversationsStore.loadConversation.bind(conversationsStore);
export const deleteConversation = conversationsStore.deleteConversation.bind(conversationsStore);
export const clearActiveConversation =
	conversationsStore.clearActiveConversation.bind(conversationsStore);
export const updateConversationName =
	conversationsStore.updateConversationName.bind(conversationsStore);
export const downloadConversation =
	conversationsStore.downloadConversation.bind(conversationsStore);
export const exportAllConversations =
	conversationsStore.exportAllConversations.bind(conversationsStore);
export const importConversations = conversationsStore.importConversations.bind(conversationsStore);
export const navigateToSibling = conversationsStore.navigateToSibling.bind(conversationsStore);
export const refreshActiveMessages =
	conversationsStore.refreshActiveMessages.bind(conversationsStore);
export const setTitleUpdateConfirmationCallback =
	conversationsStore.setTitleUpdateConfirmationCallback.bind(conversationsStore);
