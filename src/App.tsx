import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className="flex justify-center">
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="h-24 p-6 hover:drop-shadow-xl transition-transform hover:scale-110" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="h-24 p-6 hover:drop-shadow-xl transition-transform hover:scale-110 motion-safe:animate-spin" alt="React logo" />
        </a>
      </div>
      <h1 className="text-4xl font-bold mb-8">Vite + React</h1>
      <div className="p-8 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
        <button
          onClick={() => setCount((count) => count + 1)}
          className="rounded-lg border border-transparent px-5 py-2 bg-slate-900 hover:border-slate-400 transition-colors"
        >
          count is {count}
        </button>
        <p className="mt-4 text-slate-400">
          Edit <code className="font-mono bg-white/10 p-1 rounded">src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="mt-8 text-slate-400">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
