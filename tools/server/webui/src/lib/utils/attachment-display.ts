import { FileTypeCategory } from '$lib/enums';
import type { ChatAttachmentDisplayItem } from '$lib/types/chat';
import type { DatabaseMessageExtra } from '$lib/types/database';
import { isImageFile } from '$lib/utils/attachment-type';
import { getFileTypeCategory } from '$lib/utils/file-type';

export interface AttachmentDisplayItemsOptions {
	uploadedFiles?: ChatUploadedFile[];
	attachments?: DatabaseMessageExtra[];
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function getAttachmentDisplayItems(
	options: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
	const { uploadedFiles = [], attachments = [] } = options;
	const items: ChatAttachmentDisplayItem[] = [];

	// Add uploaded files (ChatForm)
	for (const file of uploadedFiles) {
		items.push({
			id: file.id,
			name: file.name,
			size: file.size,
			preview: file.preview,
			isImage: getFileTypeCategory(file.type) === FileTypeCategory.IMAGE,
			uploadedFile: file,
			textContent: file.textContent
		});
	}

	// Add stored attachments (ChatMessage)
	for (const [index, attachment] of attachments.entries()) {
		const isImage = isImageFile(attachment);

		items.push({
			id: `attachment-${index}`,
			name: attachment.name,
			preview: isImage && 'base64Url' in attachment ? attachment.base64Url : undefined,
			isImage,
			attachment,
			attachmentIndex: index,
			textContent: 'content' in attachment ? attachment.content : undefined
		});
	}

	return items.reverse();
}
