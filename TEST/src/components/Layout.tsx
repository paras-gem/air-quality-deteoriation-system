import { Outlet } from "react-router-dom";
import CursorTrail from "@/components/airq/CursorTrail";
import Navbar from "@/components/airq/Navbar";
import Footer from "@/components/airq/Footer";

export default function Layout() {
    return (
        <div className="layout-wrapper">
            <CursorTrail />
            <Navbar />
            <main className="layout-main">
                <Outlet />
            </main>
            <Footer />
        </div>
    );
}
