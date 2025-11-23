/**
 * Gets a display label for a file type
 * @param fileType - The file type/mime type
 * @returns Formatted file type label
 */
export function getFileTypeLabel(fileType: string): string {
	return fileType.split('/').pop()?.toUpperCase() || 'FILE';
}

/**
 * Truncates text content for preview display
 * @param content - The text content to truncate
 * @returns Truncated content with ellipsis if needed
 */
export function getPreviewText(content: string): string {
	return content.length > 150 ? content.substring(0, 150) + '...' : content;
}
