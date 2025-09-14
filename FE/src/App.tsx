import { useEffect, useRef, useState } from 'react'
import './App.css'
import { apiFeedback, apiPredict, type PredictMeta } from './api'
import { MoveIcon, EyeIcon, EyeOffIcon } from './components/Icons'
import { Toaster, type ToastMsg } from './components/Toast'

type Outcome = 'win' | 'draw' | 'lose'

const MOVES = ['Rock', 'Paper', 'Scissors']

function App() {
  const [uid, setUid] = useState<string>(() => `u_${Math.random().toString(36).slice(2)}`)
  const [aiMove, setAiMove] = useState<number | null>(null)
  const [meta, setMeta] = useState<PredictMeta | null>(null)
  const [history, setHistory] = useState<{ ai: number; you: number; res: Outcome }[]>([])
  const [score, setScore] = useState({ win: 0, draw: 0, lose: 0 })
  const [busy, setBusy] = useState(false)
  const [reveal, setReveal] = useState(false)
  const [toasts, setToasts] = useState<ToastMsg[]>([])
  const [roundPulse, setRoundPulse] = useState<'win'|'draw'|'lose'|'none'>('none')

  const startRound = async () => {
    setBusy(true)
    try {
      const r = await apiPredict(uid)
      setAiMove(r.ai_move)
      setMeta(r.meta)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    // prefetch first move
    startRound()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // keyboard: 1/2/3 for moves; hold 'h' to reveal AI, release to hide
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.key === '1') onPick(0)
      if (e.key === '2') onPick(1)
      if (e.key === '3') onPick(2)
      if (e.key.toLowerCase() === 'h') setReveal(true)
    }
    const onUp = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'h') setReveal(false)
    }
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp) }
  }, [aiMove])

  const onPick = async (you: number) => {
    if (aiMove == null) return
    const A = [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ]
    const resM = A[aiMove][you]
    const aiRes: Outcome = resM === 1 ? 'win' : resM === 0 ? 'draw' : 'lose' // AI perspective
    const userRes: Outcome = aiRes === 'win' ? 'lose' : aiRes === 'lose' ? 'win' : 'draw' // flip for user UI
  setHistory((h) => [{ ai: aiMove, you, res: userRes }, ...h].slice(0, 20))
  setScore((s) => ({ ...s, [userRes]: (s as any)[userRes] + 1 }))
  setRoundPulse(userRes)
  setToasts((ts) => [{ id: Date.now(), text: userRes === 'win' ? 'You win!' : userRes === 'draw' ? 'Draw' : 'AI wins', kind: userRes }, ...ts].slice(0, 3))
    await apiFeedback(uid, aiMove, you, 350, aiRes)
    setAiMove(null)
    await startRound()
  }

  const pOpp = meta?.p_opp ?? [1 / 3, 1 / 3, 1 / 3]

  return (
  <div className={`container ${roundPulse}`} onAnimationEnd={() => setRoundPulse('none')}>
      <header className="header">
        <h1>Rock · Paper · Scissors</h1>
        <div className="user">
          <label>
            User ID
            <input value={uid} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUid(e.target.value)} />
          </label>
          <button className="secondary" disabled={busy} onClick={startRound}>New Prediction</button>
          <button className="secondary" onClick={() => { setScore({ win:0, draw:0, lose:0 }); setHistory([]) }}>Reset</button>
          <button
            className="secondary"
            onMouseDown={() => setReveal(true)}
            onMouseUp={() => setReveal(false)}
            onMouseLeave={() => setReveal(false)}
            title="Hold to reveal AI (or hold H)"
          >
            {reveal ? <EyeIcon /> : <EyeOffIcon />}
          </button>
        </div>
      </header>

      <section className="scoreboard card">
        <div className="score">
          <span>Wins</span>
          <strong>{score.win}</strong>
        </div>
        <div className="score">
          <span>Draws</span>
          <strong>{score.draw}</strong>
        </div>
        <div className="score">
          <span>Losses</span>
          <strong>{score.lose}</strong>
        </div>
      </section>

      <section className="board">
        <div className="card ai">
          <h3>AI move</h3>
          <div className="move-lg">{aiMove != null ? (reveal ? <MoveIcon move={aiMove} size={72} /> : <span className="hidden">Hidden</span>) : '—'}</div>
          {meta && (
            <div className="meta">
              <span>policy: <code>{meta.policy}</code></span>
              <span>eps: {meta.eps.toFixed(2)}</span>
              <span>BR: {MOVES[meta.br_move]}</span>
            </div>
          )}
        </div>

        <div className="card dist">
          <h3>Predicted Opponent Distribution</h3>
          <div className="bars">
            {pOpp.map((v, i) => (
              <div className="bar" key={i}>
                <div className="bar-label"><MoveIcon move={i} /> {MOVES[i]}</div>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.round(v * 100)}%` }} /></div>
                <div className="bar-val">{(v * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="actions card">
        <h3>Your move</h3>
        <div className="buttons">
          {MOVES.map((m, i) => (
            <button className="action" key={m} onClick={() => onPick(i)} disabled={busy || aiMove == null}>
              <MoveIcon move={i} size={32} /> {m}
            </button>
          ))}
        </div>
        {aiMove == null && <div className="hint">Click “New Prediction” to start a round.</div>}
      </section>

      <section className="history card">
        <h3>Recent rounds</h3>
        <ul>
          {history.map((h, idx) => (
            <li key={idx} className={`row ${h.res}`}>
              <span className="you"><MoveIcon move={h.you} /> You: {MOVES[h.you]}</span>
              <span className="ai"><MoveIcon move={h.ai} /> AI: {MOVES[h.ai]}</span>
              <span className="res">{h.res === 'win' ? 'You win!' : h.res === 'draw' ? 'Draw' : 'AI wins'}</span>
            </li>
          ))}
        </ul>
      </section>

      <TwoPlayerPanel />
      <Toaster items={toasts} onDone={(id)=> setToasts((t)=> t.filter(x=>x.id!==id))} />
    </div>
  )
}

function TwoPlayerPanel() {
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<string[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  const connect = () => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (evt) => setMessages((m) => [evt.data as string, ...m].slice(0, 50))
    wsRef.current = ws
  }
  const send = (txt: string) => wsRef.current?.send(txt)

  return (
    <div className="twoplayer">
      <h3>Two-player (experimental)</h3>
      {!connected ? (
        <button onClick={connect}>Connect</button>
      ) : (
        <div>
          <button onClick={() => send('hello')}>Send hello</button>
          <div className="msgs">
            {messages.map((m, i) => (
              <div key={i}>{m}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
