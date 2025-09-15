export type PredictMeta = {
  uid: string
  p_opp: number[]
  br_move: number
  fused_move: number
  eps: number
  policy: string
  bandit: { scores: number[] }
  lookahead?: number[]
  soft_seed?: { seeded_from: string; sim: number; w: number } | null
}

const API_BASE = (import.meta as any).env?.VITE_API_BASE || '/api'

export async function apiPredict(user_hint: string | null, ctx?: any) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_hint, ctx }),
  })
  if (!res.ok) throw new Error('predict failed')
  return res.json() as Promise<{ ai_move: number; meta: PredictMeta }>
}

export async function apiFeedback(
  user_hint: string | null,
  ai_move: number,
  user_move: number,
  dt_ms: number,
  result: 'win' | 'draw' | 'lose'
) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_hint, ai_move, user_move, dt_ms, result }),
  })
  if (!res.ok) throw new Error('feedback failed')
  return res.json()
}
