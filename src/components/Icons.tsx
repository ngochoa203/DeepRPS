import React from 'react'

type IconProps = { size?: number; color?: string; className?: string; stroke?: number }

export const RockIcon: React.FC<IconProps> = ({ size = 32, color = '#93c5fd', className, stroke = 2 }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 28 L18 12 L30 10 L38 20 L34 36 L18 38 Z" stroke={color} strokeWidth={stroke} fill="none" strokeLinejoin="round"/>
  </svg>
)

export const PaperIcon: React.FC<IconProps> = ({ size = 32, color = '#86efac', className, stroke = 2 }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="12" y="8" width="24" height="32" rx="4" stroke={color} strokeWidth={stroke} fill="none"/>
    <path d="M16 16 H32 M16 22 H32 M16 28 H28" stroke={color} strokeWidth={stroke} strokeLinecap="round"/>
  </svg>
)

export const ScissorsIcon: React.FC<IconProps> = ({ size = 32, color = '#fca5a5', className, stroke = 2 }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="34" r="6" stroke={color} strokeWidth={stroke} fill="none"/>
    <circle cx="28" cy="26" r="6" stroke={color} strokeWidth={stroke} fill="none"/>
    <path d="M20 30 L42 12" stroke={color} strokeWidth={stroke} strokeLinecap="round"/>
    <path d="M20 30 L42 36" stroke={color} strokeWidth={stroke} strokeLinecap="round"/>
  </svg>
)

export const EyeIcon: React.FC<IconProps> = ({ size = 20, color = '#cbd5e1', stroke = 2, className }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" stroke={color} strokeWidth={stroke} />
    <circle cx="12" cy="12" r="3" stroke={color} strokeWidth={stroke} />
  </svg>
)

export const EyeOffIcon: React.FC<IconProps> = ({ size = 20, color = '#cbd5e1', stroke = 2, className }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 3l18 18" stroke={color} strokeWidth={stroke} strokeLinecap="round"/>
    <path d="M2 12s3.5-6 10-6c1.6 0 3 .3 4.3.8M20.9 9.8C22 10.9 22 12 22 12s-3.5 6-10 6c-2.2 0-4.1-.6-5.7-1.4" stroke={color} strokeWidth={stroke} strokeLinecap="round"/>
  </svg>
)

export const MoveIcon: React.FC<{ move: number; size?: number; className?: string }> = ({ move, size, className }) => {
  if (move === 0) return <RockIcon size={size} className={className} />
  if (move === 1) return <PaperIcon size={size} className={className} />
  return <ScissorsIcon size={size} className={className} />
}
