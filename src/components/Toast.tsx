import React, { useEffect } from 'react'

export type ToastMsg = { id: number; text: string; kind: 'win'|'draw'|'lose' }

export const Toast: React.FC<{ item: ToastMsg; onDone: (id:number)=>void }> = ({ item, onDone }) => {
  useEffect(() => {
    const t = setTimeout(() => onDone(item.id), 2000)
    return () => clearTimeout(t)
  }, [item, onDone])
  return (
    <div className={`toast ${item.kind}`}>
      {item.text}
    </div>
  )
}

export const Toaster: React.FC<{ items: ToastMsg[]; onDone:(id:number)=>void }> = ({ items, onDone }) => (
  <div className="toaster">
    {items.map(t => <Toast key={t.id} item={t} onDone={onDone} />)}
  </div>
)
