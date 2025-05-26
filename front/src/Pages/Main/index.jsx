import { Outlet } from "react-router-dom";
import { Header } from "../../Components/Header";

const Main = props => {
    return (
        <div className="w-full flex flex-wrap">
            <Header />
            <Outlet />
        </div>
    )
}

export { Main };