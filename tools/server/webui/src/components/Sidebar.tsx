import { useEffect, useState } from 'react';
import { classNames } from '../utils/misc';
import { Conversation } from '../utils/types';
import StorageUtils from '../utils/storage';
import { useNavigate, useParams } from 'react-router';
import {
  ArrowDownTrayIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { BtnWithTooltips } from '../utils/common';
import { useAppContext } from '../utils/app.context';
import toast from 'react-hot-toast';

export default function Sidebar() {
  const params = useParams();
  const navigate = useNavigate();

  const { isGenerating } = useAppContext();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currConv, setCurrConv] = useState<Conversation | null>(null);

  useEffect(() => {
    StorageUtils.getOneConversation(params.convId ?? '').then(setCurrConv);
  }, [params.convId]);

  useEffect(() => {
    const handleConversationChange = async () => {
      setConversations(await StorageUtils.getAllConversations());
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    handleConversationChange();
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
  }, []);

  return (
    <>
      <input
        id="toggle-drawer"
        type="checkbox"
        className="drawer-toggle"
        defaultChecked
      />

      <div className="drawer-side h-screen lg:h-screen z-50 lg:max-w-64">
        <label
          htmlFor="toggle-drawer"
          aria-label="close sidebar"
          className="drawer-overlay"
        ></label>
        <div className="flex flex-col bg-base-200 min-h-full max-w-64 py-4 px-4">
          <div className="flex flex-row items-center justify-between mb-4 mt-4">
            <h2 className="font-bold ml-4">Conversations</h2>

            {/* close sidebar button */}
            <label htmlFor="toggle-drawer" className="btn btn-ghost lg:hidden">
              <XMarkIcon className="w-5 h-5" />
            </label>
          </div>

          {/* list of conversations */}
          <div
            className={classNames({
              'btn btn-ghost justify-start': true,
              'btn-soft': !currConv,
            })}
            onClick={() => navigate('/')}
          >
            + New conversation
          </div>
          {conversations.map((conv) => (
            <ConversationItem
              key={conv.id}
              conv={conv}
              isCurrConv={currConv?.id === conv.id}
              onSelect={() => {
                navigate(`/chat/${conv.id}`);
              }}
              onDelete={() => {
                if (isGenerating(conv.id)) {
                  toast.error('Cannot delete conversation while generating');
                  return;
                }
                if (
                  window.confirm('Are you sure to delete this conversation?')
                ) {
                  toast.success('Conversation deleted');
                  StorageUtils.remove(conv.id);
                  navigate('/');
                }
              }}
              onDownload={() => {
                if (isGenerating(conv.id)) {
                  toast.error('Cannot download conversation while generating');
                  return;
                }
                const conversationJson = JSON.stringify(conv, null, 2);
                const blob = new Blob([conversationJson], {
                  type: 'application/json',
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `conversation_${conv.id}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
              onRename={() => {
                if (isGenerating(conv.id)) {
                  toast.error('Cannot rename conversation while generating');
                  return;
                }
                const newName = window.prompt(
                  'Enter new name for the conversation',
                  conv.name
                );
                if (newName && newName.trim().length > 0) {
                  StorageUtils.updateConversationName(conv.id, newName);
                }
              }}
            />
          ))}
          <div className="text-center text-xs opacity-40 mt-auto mx-4">
            Conversations are saved to browser's IndexedDB
          </div>
        </div>
      </div>
    </>
  );
}

function ConversationItem({
  conv,
  isCurrConv,
  onSelect,
  onDelete,
  onDownload,
  onRename,
}: {
  conv: Conversation;
  isCurrConv: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onDownload: () => void;
  onRename: () => void;
}) {
  return (
    <div
      className={classNames({
        'group flex flex-row btn btn-ghost justify-start items-center font-normal pr-2':
          true,
        'btn-soft': isCurrConv,
      })}
    >
      <div
        key={conv.id}
        className="w-full overflow-hidden truncate text-start"
        onClick={onSelect}
        dir="auto"
      >
        {conv.name}
      </div>
      <div className="dropdown dropdown-end h-5">
        <BtnWithTooltips
          // on mobile, we always show the ellipsis icon
          // on desktop, we only show it when the user hovers over the conversation item
          // we use opacity instead of hidden to avoid layout shift
          className="cursor-pointer opacity-100 md:opacity-0 group-hover:opacity-100"
          onClick={() => {}}
          tooltipsContent="More"
        >
          <EllipsisVerticalIcon className="w-5 h-5" />
        </BtnWithTooltips>
        {/* dropdown menu */}
        <ul
          tabIndex={0}
          className="dropdown-content menu bg-base-100 rounded-box z-[1] p-2 shadow"
        >
          <li onClick={onRename}>
            <a>
              <PencilIcon className="w-4 h-4" />
              Rename
            </a>
          </li>
          <li onClick={onDownload}>
            <a>
              <ArrowDownTrayIcon className="w-4 h-4" />
              Download
            </a>
          </li>
          <li className="text-error" onClick={onDelete}>
            <a>
              <TrashIcon className="w-4 h-4" />
              Delete
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
}
