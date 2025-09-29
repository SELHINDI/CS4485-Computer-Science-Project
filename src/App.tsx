import { Outlet, RouterProvider, createBrowserRouter, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Multi from './pages/Multi'
import Explorer from './pages/Explorer'
import Metrics from './pages/Metrics'
import { Navbar } from './app/components/Navbar'

function Layout() {
  return (
    <div className="min-h-dvh flex flex-col">
      <Navbar />
      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-4 py-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'multi', element: <Multi /> },
      { path: 'explorer', element: <Explorer /> },
      { path: 'metrics', element: <Metrics /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
