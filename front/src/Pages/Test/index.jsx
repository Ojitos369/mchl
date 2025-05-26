import { useStates } from '../../Hooks/useStates';
import styles from './styles/index.module.scss';

export const Test = props => {
    const { ls, lf, s, f } = useStates();
    return (
        <div className={`${styles.testPage}`}>
            Component to make tests
        </div>
    )
}

