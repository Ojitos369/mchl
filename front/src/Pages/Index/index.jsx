import { useMemo, useEffect } from 'react';
import { useStates } from '../../Hooks/useStates';
import { Link } from "react-router-dom";
import styles from './styles/index.module.scss';

const Index = props => {
    const { s, f } = useStates();

    const apps = [
        {label: "Puertas Logicas", to: "puertas_logicas"},
        {label: "Escritura Numero", to: "escritura_numero"},
        {label: "Escritura Letra", to: "escritura_letra"},
    ]

    useEffect(() => {
        f.u1("page", "title", "Index");
    }, []);

    return (
        <div className={` ${styles.indexPage} flex flex-row w-full flex-wrap justify-around mt-4`}>
            {apps.map((app, i) => {
                const show = app.show ?? true;
                if (!show) return null;
                return (
                    <div className={`${styles.linkElement}`}>
                        <Link to={app.to}
                            key={i}
                            className={`${styles.linkContent}`}
                            >
                            {app.label}
                        </Link>
                    </div>
                )
            })}
        </div>
    )
}

export { Index };
