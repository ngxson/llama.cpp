import { useState } from 'react';
import { MessageExtra } from '../utils/types';
import toast from 'react-hot-toast';

// Interface describing the API returned by the hook
export interface ChatExtraContextApi {
  items?: MessageExtra[]; // undefined if empty, similar to Message['extra']
  addItems: (items: MessageExtra[]) => void;
  removeItem: (idx: number) => void;
  clearItems: () => void;
  onFileAdded: (files: File[]) => void; // used by "upload" button
}

export function useChatExtraContext(): ChatExtraContextApi {
  const [items, setItems] = useState<MessageExtra[]>([]);

  const addItems = (newItems: MessageExtra[]) => {
    setItems((prev) => [...prev, ...newItems]);
  };

  const removeItem = (idx: number) => {
    setItems((prev) => prev.filter((_, i) => i !== idx));
  };

  const clearItems = () => {
    setItems([]);
  };

  const onFileAdded = (files: File[]) => {
    for (const file of files) {
      const mimeType = file.type;
      console.debug({ mimeType, file });
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File is too large. Maximum size is 10MB.');
        break;
      }
      if (mimeType.startsWith('text/')) {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            addItems([
              {
                type: 'textFile',
                name: file.name,
                content: event.target.result as string,
              },
            ]);
          }
        };
        reader.readAsText(file);
      } else if (mimeType.startsWith('image/')) {
        // TODO @ngxson : throw an error if the server does not support image input
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            addItems([
              {
                type: 'imageFile',
                name: file.name,
                base64Url: event.target.result as string,
              },
            ]);
          }
        };
        reader.readAsDataURL(file);
      } else {
        // TODO @ngxson : support all other file formats like .pdf, .py, .bat, .c, etc
        toast.error('Unsupported file type.');
      }
    }
  };

  return {
    items: items.length > 0 ? items : undefined,
    addItems,
    removeItem,
    clearItems,
    onFileAdded,
  };
}
