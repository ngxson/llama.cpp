import { AttachmentType } from '$lib/enums/attachment';
import { FileTypeCategory } from '$lib/enums/files';
import { getFileTypeCategory } from '$lib/utils/file-type';
import { getFileTypeLabel } from '$lib/utils/file-preview';
import type { DatabaseMessageExtra } from '$lib/types/database';

/**
 * Determines if an attachment or uploaded file is a text file
 * @param uploadedFile - Optional uploaded file
 * @param attachment - Optional database attachment
 * @returns true if the file is a text file
 */
export function isTextFile(
	attachment?: DatabaseMessageExtra,
	uploadedFile?: ChatUploadedFile
): boolean {
	if (uploadedFile) {
		return getFileTypeCategory(uploadedFile.type) === FileTypeCategory.TEXT;
	}

	if (attachment) {
		return (
			attachment.type === AttachmentType.TEXT || attachment.type === AttachmentType.LEGACY_CONTEXT
		);
	}

	return false;
}

/**
 * Determines if an attachment or uploaded file is an image
 * @param uploadedFile - Optional uploaded file
 * @param attachment - Optional database attachment
 * @returns true if the file is an image
 */
export function isImageFile(
	attachment?: DatabaseMessageExtra,
	uploadedFile?: ChatUploadedFile
): boolean {
	if (uploadedFile) {
		return getFileTypeCategory(uploadedFile.type) === FileTypeCategory.IMAGE;
	}

	if (attachment) {
		return attachment.type === AttachmentType.IMAGE;
	}

	return false;
}

/**
 * Determines if an attachment or uploaded file is a PDF
 * @param uploadedFile - Optional uploaded file
 * @param attachment - Optional database attachment
 * @returns true if the file is a PDF
 */
export function isPdfFile(
	attachment?: DatabaseMessageExtra,
	uploadedFile?: ChatUploadedFile
): boolean {
	if (uploadedFile) {
		return uploadedFile.type === 'application/pdf';
	}

	if (attachment) {
		return attachment.type === AttachmentType.PDF;
	}

	return false;
}

/**
 * Determines if an attachment or uploaded file is an audio file
 * @param uploadedFile - Optional uploaded file
 * @param attachment - Optional database attachment
 * @returns true if the file is an audio file
 */
export function isAudioFile(
	attachment?: DatabaseMessageExtra,
	uploadedFile?: ChatUploadedFile
): boolean {
	if (uploadedFile) {
		return getFileTypeCategory(uploadedFile.type) === FileTypeCategory.AUDIO;
	}

	if (attachment) {
		return attachment.type === AttachmentType.AUDIO;
	}

	return false;
}

/**
 * Gets a human-readable type label for display
 * @param uploadedFile - Optional uploaded file
 * @param attachment - Optional database attachment
 * @returns A formatted type label string
 */
export function getAttachmentTypeLabel(
	attachment?: DatabaseMessageExtra,
	uploadedFile?: ChatUploadedFile
): string {
	if (uploadedFile) {
		// For uploaded files, use the file type label utility
		return getFileTypeLabel(uploadedFile.type);
	}

	if (attachment) {
		// For attachments, convert enum to readable format
		switch (attachment.type) {
			case AttachmentType.IMAGE:
				return 'image';
			case AttachmentType.AUDIO:
				return 'audio';
			case AttachmentType.PDF:
				return 'pdf';
			case AttachmentType.TEXT:
			case AttachmentType.LEGACY_CONTEXT:
				return 'text';
			default:
				return 'unknown';
		}
	}

	return 'unknown';
}
