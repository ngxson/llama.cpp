import { useEffect, useState } from 'react';
import { classNames } from '../utils/misc';
import { Conversation } from '../utils/types';
import StorageUtils from '../utils/storage';
import { useNavigate, useParams } from 'react-router';
import { XMarkIcon } from '@heroicons/react/24/outline';

export default function Sidebar() {
  const params = useParams();
  const navigate = useNavigate();

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
              'btn-active': !currConv,
            })}
            onClick={() => navigate('/')}
          >
            + New conversation
          </div>
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={classNames({
                'btn btn-ghost justify-start font-normal': true,
                'btn-active': conv.id === currConv?.id,
              })}
              onClick={() => navigate(`/chat/${conv.id}`)}
              dir="auto"
            >
              <span className="truncate">{conv.name}</span>
            </div>
          ))}
          <div className="text-center text-xs opacity-40 mt-auto mx-4">
            Conversations are saved to browser's IndexedDB
          </div>
        </div>
      </div>
    </>
  );
}
