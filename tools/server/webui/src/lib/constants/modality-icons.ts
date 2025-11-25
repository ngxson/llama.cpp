/**
 * Icon mappings for file types and model modalities
 * Centralized configuration to ensure consistent icon usage across the app
 */

import {
	File as FileIcon,
	FileText as FileTextIcon,
	Image as ImageIcon,
	Volume2 as AudioIcon
} from '@lucide/svelte';
import { FileTypeCategory } from '$lib/enums';

export const FILE_TYPE_ICONS = {
	[FileTypeCategory.IMAGE]: ImageIcon,
	[FileTypeCategory.AUDIO]: AudioIcon,
	[FileTypeCategory.TEXT]: FileTextIcon,
	[FileTypeCategory.PDF]: FileIcon
} as const;

export const DEFAULT_FILE_ICON = FileIcon;

export type ModelModality = 'vision' | 'audio';

export const MODALITY_ICONS = {
	vision: ImageIcon,
	audio: AudioIcon
} as const;
