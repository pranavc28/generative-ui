# React Viewer (TypeScript Enabled)

A React + TypeScript development environment to run and view the model-generated React code.

## Setup (One-time - Already Done!)

```bash
npm install
```

## Usage

1. **Start the development server:**
   ```bash
   npm run dev
   ```
   OR
   ```bash
   ./START.sh
   ```
   This will open a browser at http://localhost:5173

2. **Copy-paste React/TypeScript code:**
   - Open one of the HTML viewer pages (in ../outputs/)
   - Copy the React code (including interfaces, types, etc.)
   - Paste it into `src/App.tsx` (replace everything)
   - Save the file

3. **See it live:**
   - The page will automatically reload with your code running!

## ✨ Features

- ✅ **TypeScript support** - interfaces, types, all work!
- ✅ **Hot reload** - changes appear instantly
- ✅ **TailwindCSS** already included
- ✅ **React 18** with latest Vite
- ✅ **No configuration needed** - just paste and run!

## Notes

- Files are now `.tsx` (TypeScript + JSX)
- TypeScript is configured with relaxed rules for easy pasting
- To stop the server: Press `Ctrl+C` in the terminal

## Example

Replace the entire contents of `src/App.tsx` with any of the model outputs:
- Example 0: Easy Rider Transportation
- Example 1: Google Homepage  
- Example 2: Realtime Leaderboard

Just copy the entire code block and paste it in!
