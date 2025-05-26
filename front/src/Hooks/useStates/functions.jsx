import axios from "axios";
import Swal from 'sweetalert2';
import withReactContent from 'sweetalert2-react-content';
import { useDispatch, useSelector } from "react-redux";
import { f as ff } from "./fs";


const MySwal = withReactContent(Swal);

const link = 'http://localhost:8369/api/';
axios.defaults.withCredentials = true
const miAxios = axios.create({
    baseURL: link,
});


const useF = props => {
    const ls = useSelector(state => state.fs.ls);
    const s = useSelector(state => state.fs.s);
    const d = useDispatch();

    const app = {
        helloWorld: () => {
            const end = 'app/hello_world/';
            miAxios.get(end)
            .then(res => {
                console.log(res.data);
            })
            .catch(err => {
                console.log(err);
            });
        }
    }

    const puertas_logicas = {
        train: (x, y, lr, af) => {
            if (!!s.puertas_logicas?.training) return;
            u1("puertas_logicas", "training", true);
            u1("puertas_logicas", "trained", false);
            u1("puertas_logicas", "respuesta", "");
            
            const end = 'puertas_logicas/train/';
            const data = {x, y, lr, af};
            u1("puertas_logicas", "resumen", data);
            miAxios.post(end, data)
            .then(res => {
                const { message } = res.data;
                general.notificacion({
                    message, 
                    mode: "success"
                });
                u1("puertas_logicas", "trained", true);
            })
            .catch(err => {
                const { message } = err.response.data;
                general.notificacion({
                    message, 
                    mode: "danger"
                 });
                 u1("puertas_logicas", "trained", false);
            }).finally(() => {
                u1("puertas_logicas", "training", false);
            });
        },
        calculate: (x1, x2, af) => {
            // const data = {x1, x2};
            const end = `puertas_logicas/calculate/?x1=${x1}&x2=${x2}&af=${af}`;
            miAxios.get(end)
            .then(res => {
                const { respuesta } = res.data;
                u1("puertas_logicas", "respuesta", respuesta);
            })
            .catch(err => {
                const { message } = err.response.data;
                general.notificacion({
                    message, 
                    mode: "danger"
                 });
            });
        }
    }

    const general = {
        notificacion: props => {
            u1("general", "notification", props);
            u2("modals", "general", "notification", true);
        }
    }

    // u[0-9]
    const u0 = (f0, value) => {
        d(ff.u0({f0, value}));
    }
    const u1 = (f0, f1, value) => {
        d(ff.u1({f0, f1, value}));
    }
    const u2 = (f0, f1, f2, value) => {
        d(ff.u2({f0, f1, f2, value}));
    }
    const u3 = (f0, f1, f2, f3, value) => {
        d(ff.u3({f0, f1, f2, f3, value}));
    }
    const u4 = (f0, f1, f2, f3, f4, value) => {
        d(ff.u4({f0, f1, f2, f3, f4, value}));
    }
    const u5 = (f0, f1, f2, f3, f4, f5, value) => {
        d(ff.u5({f0, f1, f2, f3, f4, f5, value}));
    }
    const u6 = (f0, f1, f2, f3, f4, f5, f6, value) => {
        d(ff.u6({f0, f1, f2, f3, f4, f5, f6, value}));
    }
    const u7 = (f0, f1, f2, f3, f4, f5, f6, f7, value) => {
        d(ff.u7({f0, f1, f2, f3, f4, f5, f6, f7, value}));
    }
    const u8 = (f0, f1, f2, f3, f4, f5, f6, f7, f8, value) => {
        d(ff.u8({f0, f1, f2, f3, f4, f5, f6, f7, f8, value}));
    }
    const u9 = (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, value) => {
        d(ff.u9({f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, value}));
    }

    return { 
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, 
        app, puertas_logicas, general, 
    };
}

export { useF };