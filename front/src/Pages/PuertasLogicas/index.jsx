import { useEffect, useMemo, useState } from 'react';
import { useStates } from '../../Hooks/useStates';
import styles from './styles/index.module.scss';

export const PuertasLogicas = props => {
    const { s, f } = useStates();
    const trained = useMemo(() => s.puertas_logicas?.trained ?? true, [s.puertas_logicas?.trained]);

    return (
        <div className={`${styles.plPage}`}>
            <TrainRow />
            {trained ? <TestRow /> : <Resumen />}
        </div>
    )
}


const TrainRow = props => {
    const { s, f } = useStates();
    const [x, setX] = useState([[0, 0, 1, 1], [0, 1, 0, 1]]);
    const [y, setY] = useState([[0, 0, 0, 1]]);
    const [mode, setMode] = useState("and");
    const [lr, setLr] = useState(0.5);
    const [af, setAf] = useState("sig");
    const [pasos, setPasos] = useState(500000);
    const training = useMemo(() => s.puertas_logicas?.training, [s.puertas_logicas?.training]);

    const changeMode = mode => {
        setMode(mode);
    }
    const changeAf = mode => {
        setAf(mode);
    }
    const changeLr = e => {
        setLr(e.target.value);
    }
    const changePasos = e => {
        setPasos(e.target.value);
    }

    const train = e => {
        if (!!e) e.preventDefault();
        console.log("Y", y);
        f.puertas_logicas.train(x, y, lr, af);
    }

    useEffect(() => {
        f.u1("page", "title", "Puertas Logicas");
        changeMode("and");
    }, []);

    useEffect(() => {
        switch (mode) {
            case "and":
                setY([[0, 0, 0, 1]]);
                break;
            case "or":
                setY([[0, 1, 1, 1]]);
                break;
            case "xor":
                setY([[0, 1, 1, 0]]);
                break;
            default:
                setY(null);
                break;
        }
    }, [mode]);

    return (
        <div className={`${styles.trainRow}`}>
            <div className={`${styles.selectMode}`}>
                <label className={`${styles.label}`} >Modo</label>
                <select 
                    className={`${styles.select}`}
                    value={mode}
                    onChange={e => changeMode(e.target.value)}
                    >
                    <option value="and">AND</option>
                    <option value="or">OR</option>
                    <option value="xor">XOR</option>
                </select>
            </div>
            <div className={`${styles.selectMode}`}>
                <label className={`${styles.label}`} >Funcion</label>
                <select 
                    className={`${styles.select}`}
                    onChange={e => changeAf(e.target.value)}>
                    <option value="sig">Sigmoide</option>
                    <option value="tanh">Tangente Hiperbolica</option>
                </select>
            </div>
            <div className={`${styles.selectMode}`}>
                <label className={`${styles.label}`} >Learning Rate</label>
                <input 
                    className={`${styles.select}`}
                    value={lr}
                    onChange={changeLr}
                    />
            </div>
            <div className={`${styles.selectMode}`}>
                <label className={`${styles.label}`} >Pasos</label>
                <input 
                    className={`${styles.select}`}
                    value={pasos}
                    onChange={changePasos}
                    />
            </div>

            <div className={`${styles.trainButton}`}>
                <button className={` ${styles.button} ${training ? styles.disbled : ""}`} onClick={train}>
                    {training ?
                    'Training' :
                    'Train'}
                </button>
            </div>
        </div>
    )
}

const TestRow = props => {
    const { s, f } = useStates();
    const [x1, setX1] = useState(0);
    const [x2, setX2] = useState(0);
    const loading = useMemo(() => s.puertas_logicas?.loading, [s.puertas_logicas?.loading]);
    const respuesta = useMemo(() => s.puertas_logicas?.respuesta, [s.puertas_logicas?.respuesta]);
    const {x, y, lr, af} = useMemo(() => s.puertas_logicas?.resumen || {}, [s.puertas_logicas?.resumen]);

    const toggleInput = (mode) => {
        if (mode === 1) {
            setX1(x1 === 0 ? 1 : 0);
        } else if (mode === 2) {
            setX2(x2 === 0 ? 1 : 0);
        }
    }

    useEffect(() => {
        f.puertas_logicas.calculate(x1, x2, af);
    }, [x1, x2]);

    return (
        <div className={`${styles.testRow}`}>
            <div className={`${styles.inputsTestRow}`}>
                <div className={`${styles.elementInput}`}>
                    <label className={`${styles.label}`} >
                        X1
                    </label>
                    <button 
                        className={`${styles.toggle} ${x1 === 1 ? styles.toggleOn : styles.toggleOff}`}
                        onClick={() => toggleInput(1)}
                        >
                        <span className={`${styles.value}`}>
                            { x1 }
                        </span>
                    </button>
                </div>
                <div className={`${styles.elementInput}`}>
                    <label className={`${styles.label}`} >
                        X2
                    </label>
                    <button 
                        className={`${styles.toggle} ${x2 === 1 ? styles.toggleOn : styles.toggleOff}`}
                        onClick={() => toggleInput(2)}
                        >
                        <span className={`${styles.value}`}>
                            { x2 }
                        </span>
                    </button>
                </div>

                <div className={`${styles.respuesta} w-full`}>
                    <p className={`${styles.label}`} >
                        R: {respuesta}
                    </p>
                </div>
            </div>
        </div>
    )
}

const Resumen = props => {
    const { s } = useStates();
    const {x, y, lr, af} = useMemo(() => s.puertas_logicas?.resumen || {}, [s.puertas_logicas?.resumen]);

    return (
        <div className={`${styles.resumen}`}>
            <h2 className={`${styles.title}`}>Entrenando con los siguientes Datos: </h2>
            <div className={`${styles.details}`}>
                <p className={`${styles.detail}`}>
                    <strong>Entradas (X):</strong> {JSON.stringify(x)}
                </p>
                <p className={`${styles.detail}`}>
                    <strong>Salidas Esperadas (Y):</strong> {JSON.stringify(y)}
                </p>
                <p className={`${styles.detail}`}>
                    <strong>Learning Rate:</strong> {lr}
                </p>
                <p className={`${styles.detail}`}>
                    <strong>Función de Activación:</strong> {af}
                </p>
            </div>
        </div>
    )
}