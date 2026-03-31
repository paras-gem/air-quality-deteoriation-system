import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import PredictPage from "./pages/PredictPage";
import HealthPage from "./pages/HealthPage";
import AlertsPage from "./pages/AlertsPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Layout />}>
                    <Route index element={<HomePage />} />
                    <Route path="predict" element={<PredictPage />} />
                    <Route path="health" element={<HealthPage />} />
                    <Route path="alerts" element={<AlertsPage />} />
                    <Route path="about" element={<AboutPage />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}
