import { useMemo } from "react";
import { useStates } from "../../Hooks/useStates";
import { Link } from "react-router-dom";
import styles from './styles/index.module.scss';

export const Header = props => {
    const { s } = useStates();
    const title = useMemo(() => s.page?.title ?? '', [s.page?.title]);

    return (
        <div className={` ${styles.header} flex w-full bg-[var(--my-primary)] h-8 items-center px-6`}>
            <span className="underline">
                <Link to='/' className="underline">
                        Inicio
                </Link>
            </span>
            <span className="mx-3">
                /
            </span>
            <span>
                {title}
            </span>
        </div>
    )
}
