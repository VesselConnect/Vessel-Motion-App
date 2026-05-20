import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title='Vessel Motion Calculator', layout='wide')

G = 9.81

if 'single_result_df' not in st.session_state:
    st.session_state.single_result_df = None
if 'grid_result_df' not in st.session_state:
    st.session_state.grid_result_df = None
if 'plot_metric' not in st.session_state:
    st.session_state.plot_metric = 'RMS vertical displacement (m)'


def norm_phase_deg(phi_deg):
    return np.mod(phi_deg, 360.0)


def excel_col_name(idx):
    idx += 1
    name = ''
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def find_header_rows(raw, search_rows=8):
    motion_row = None
    subheader_row = None
    period_row = None
    for r in range(min(search_rows, raw.shape[0])):
        vals = [str(raw.iat[r, c]).strip().lower() if pd.notna(raw.iat[r, c]) else '' for c in range(raw.shape[1])]
        if 'wave period' in vals:
            period_row = r
        if any(v in ['heave', 'roll', 'pitch'] for v in vals):
            motion_row = r
        if any(v in ['amplitude', 'phase'] for v in vals):
            subheader_row = r
    if motion_row is None or subheader_row is None or period_row is None:
        raise ValueError('Could not identify header rows')
    return period_row, motion_row, subheader_row


def detect_columns(raw):
    period_row, motion_row, subheader_row = find_header_rows(raw)
    ncols = raw.shape[1]
    period_col = None
    for c in range(ncols):
        v = str(raw.iat[period_row, c]).strip().lower() if pd.notna(raw.iat[period_row, c]) else ''
        if v == 'wave period':
            period_col = c
            break
    if period_col is None:
        raise ValueError('Could not find Wave Period column')

    detected = {
        'period_col': period_col,
        'period_row': period_row,
        'motion_row': motion_row,
        'subheader_row': subheader_row,
        'period_excel': excel_col_name(period_col),
    }

    for motion in ['heave', 'roll', 'pitch']:
        amp_col = None
        phase_col = None
        for c in range(ncols):
            mv = str(raw.iat[motion_row, c]).strip().lower() if pd.notna(raw.iat[motion_row, c]) else ''
            sv = str(raw.iat[subheader_row, c]).strip().lower() if pd.notna(raw.iat[subheader_row, c]) else ''
            if mv == motion and sv == 'amplitude' and amp_col is None:
                amp_col = c
            if mv == motion and sv == 'phase' and phase_col is None:
                phase_col = c
        if amp_col is None or phase_col is None:
            motion_start = None
            for c in range(ncols):
                mv = str(raw.iat[motion_row, c]).strip().lower() if pd.notna(raw.iat[motion_row, c]) else ''
                if mv == motion:
                    motion_start = c
                    break
            if motion_start is not None:
                for c in range(motion_start, min(motion_start + 4, ncols)):
                    sv = str(raw.iat[subheader_row, c]).strip().lower() if pd.notna(raw.iat[subheader_row, c]) else ''
                    if sv == 'amplitude' and amp_col is None:
                        amp_col = c
                    if sv == 'phase' and phase_col is None:
                        phase_col = c
        if amp_col is None or phase_col is None:
            raise ValueError(f'Could not detect amplitude/phase columns for {motion}')
        detected[f'{motion}_amp_col'] = amp_col
        detected[f'{motion}_phase_col'] = phase_col
        detected[f'{motion}_amp_excel'] = excel_col_name(amp_col)
        detected[f'{motion}_phase_excel'] = excel_col_name(phase_col)
    return detected


@st.cache_data
def get_sheet_names(file_path):
    return pd.ExcelFile(file_path).sheet_names


@st.cache_data
def load_raw_sheet(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name, header=None)


def parse_sheet(raw, cols):
    data_start_row = cols['subheader_row'] + 2
    selected = raw.iloc[data_start_row:, [
        cols['period_col'],
        cols['heave_amp_col'], cols['heave_phase_col'],
        cols['roll_amp_col'], cols['roll_phase_col'],
        cols['pitch_amp_col'], cols['pitch_phase_col'],
    ]].copy()
    selected.columns = [
        'T',
        'heave_amp', 'heave_phase_deg',
        'roll_amp_deg_per_m', 'roll_phase_deg',
        'pitch_amp_deg_per_m', 'pitch_phase_deg',
    ]
    for col in selected.columns:
        selected[col] = pd.to_numeric(selected[col], errors='coerce')
    selected = selected.dropna()
    selected = selected[selected['T'] > 0].copy()
    selected = selected.sort_values('T').reset_index(drop=True)
    if len(selected) < 5:
        raise ValueError('Too few valid RAO rows found after parsing')
    return selected


def jonswap_shape(omega, Tp, gamma=3.3):
    wp = 2.0 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2))
    base = (G ** 2) * (omega ** -5.0) * np.exp(-1.25 * (wp / omega) ** 4.0)
    return base * (gamma ** r)


def scale_spectrum_to_hs(omega, S0, hs_target):
    m0_0 = np.trapz(S0, omega)
    hs_0 = 4.0 * np.sqrt(m0_0)
    scale = (hs_target / hs_0) ** 2.0
    return scale * S0


def build_point_vertical_rao(df, x_point, y_point):
    T = df['T'].to_numpy(dtype=float)
    omega = 2.0 * np.pi / T
    heave_amp = df['heave_amp'].to_numpy(dtype=float)
    heave_phase = np.deg2rad(norm_phase_deg(df['heave_phase_deg'].to_numpy(dtype=float)))
    roll_amp = df['roll_amp_deg_per_m'].to_numpy(dtype=float)
    roll_phase = np.deg2rad(norm_phase_deg(df['roll_phase_deg'].to_numpy(dtype=float)))
    pitch_amp = df['pitch_amp_deg_per_m'].to_numpy(dtype=float)
    pitch_phase = np.deg2rad(norm_phase_deg(df['pitch_phase_deg'].to_numpy(dtype=float)))
    rao_heave = heave_amp * np.exp(1j * heave_phase)
    rao_roll = (roll_amp * np.pi / 180.0) * np.exp(1j * roll_phase)
    rao_pitch = (pitch_amp * np.pi / 180.0) * np.exp(1j * pitch_phase)
    rao_point = rao_heave - x_point * rao_pitch + y_point * rao_roll
    order = np.argsort(omega)
    return omega[order], rao_point[order]


def most_probable_maximum(sigma, Tz, duration_seconds):
    if sigma <= 0 or Tz <= 0 or duration_seconds <= 0:
        return np.nan
    n = max(duration_seconds / Tz, 1.0)
    if n <= 1.0:
        return sigma
    return sigma * np.sqrt(2.0 * np.log(n))


def compute_response(omega, rao_point, hs, tp, gamma, duration_hours):
    S0 = jonswap_shape(omega, tp, gamma)
    S_eta = scale_spectrum_to_hs(omega, S0, hs)
    S_z = (np.abs(rao_point) ** 2) * S_eta
    S_v = (omega ** 2) * S_z
    S_a = (omega ** 4) * S_z

    m0_z = np.trapz(S_z, omega)
    m2_z = np.trapz((omega ** 2) * S_z, omega)
    m0_v = np.trapz(S_v, omega)
    m2_v = np.trapz((omega ** 2) * S_v, omega)
    m0_a = np.trapz(S_a, omega)

    rms_z = np.sqrt(m0_z)
    rms_v = np.sqrt(m0_v)
    rms_a = np.sqrt(m0_a)

    Tz_z = 2.0 * np.pi * np.sqrt(m0_z / m2_z) if m0_z > 0 and m2_z > 0 else np.nan
    Tz_v = 2.0 * np.pi * np.sqrt(m0_v / m2_v) if m0_v > 0 and m2_v > 0 else np.nan

    duration_seconds = duration_hours * 3600.0
    mpm_z = most_probable_maximum(rms_z, Tz_z, duration_seconds)
    mpm_v = most_probable_maximum(rms_v, Tz_v, duration_seconds)

    return {
        'RMS vertical displacement (m)': rms_z,
        'Significant vertical displacement (m)': 2.0 * rms_z,
        'Most probable max vertical displacement (m)': mpm_z,
        'RMS vertical velocity (m/s)': rms_v,
        'Significant vertical velocity (m/s)': 2.0 * rms_v,
        'Most probable max vertical velocity (m/s)': mpm_v,
        'RMS vertical acceleration (m/s²)': rms_a,
        'Significant vertical acceleration (m/s²)': 2.0 * rms_a,
        'Response Tz displacement (s)': Tz_z,
        'Response Tz velocity (s)': Tz_v,
    }


def parse_range(text):
    values = []
    for part in text.split(','):
        p = part.strip()
        if p:
            values.append(float(p))
    return values


st.title('Vessel Point Motion Calculator')
st.write('Corrected parser version with most probable maximum vertical displacement and velocity over a selected duration.')

uploaded_file = st.file_uploader('Upload RAO Excel workbook', type=['xlsx'])

if uploaded_file is not None:
    temp_path = Path('/tmp/rao_input_corrected_extremes.xlsx')
    temp_path.write_bytes(uploaded_file.read())

    try:
        sheet_names = get_sheet_names(temp_path)
        default_idx = sheet_names.index('Heading_180 deg') if 'Heading_180 deg' in sheet_names else 0
        sheet_name = st.selectbox('Heading sheet', sheet_names, index=default_idx)

        raw = load_raw_sheet(temp_path, sheet_name)
        cols = detect_columns(raw)

        with st.expander('Parser diagnostics'):
            st.json(cols)
            st.dataframe(raw.iloc[:8, :14], use_container_width=True)
            st.dataframe(parse_sheet(raw, cols).head(15), use_container_width=True)

        st.subheader('Point coordinates')
        c1, c2, c3 = st.columns(3)
        with c1:
            x_point = st.number_input('X coordinate (m)', value=-10.0, step=1.0)
        with c2:
            y_point = st.number_input('Y coordinate (m)', value=-9.0, step=1.0)
        with c3:
            z_point = st.number_input('Z coordinate (m)', value=4.0, step=1.0)

        st.caption('Z is stored for reference, but the vertical rigid-body displacement formula uses heave, pitch with X offset, and roll with Y offset.')

        st.subheader('Sea state input')
        mode = st.radio('Mode', ['Single sea state', 'Sea-state grid'], horizontal=True)
        gamma = st.number_input('JONSWAP gamma', value=3.3, min_value=1.0, step=0.1)
        duration_hours = st.number_input('Duration for most probable maximum (hours)', value=3.0, min_value=0.1, step=0.5)

        df = parse_sheet(raw, cols)
        omega, rao_point = build_point_vertical_rao(df, x_point, y_point)

        if mode == 'Single sea state':
            c4, c5 = st.columns(2)
            with c4:
                hs = st.number_input('Hs (m)', value=2.0, min_value=0.0, step=0.1)
            with c5:
                tp = st.number_input('Tp (s)', value=10.0, min_value=0.1, step=0.5)

            if st.button('Calculate single sea state'):
                result = compute_response(omega, rao_point, hs, tp, gamma, duration_hours)
                result_df = pd.DataFrame({'Metric': list(result.keys()), 'Value': list(result.values())})
                result_df['Value'] = result_df['Value'].round(4)
                st.session_state.single_result_df = result_df
                st.session_state.grid_result_df = None

            if st.session_state.single_result_df is not None:
                st.subheader('Results')
                st.dataframe(st.session_state.single_result_df, use_container_width=True)
        else:
            st.write('Enter comma-separated values, for example Hs: 1.5,2.0 and Tp: 8,9,10,11,12,13')
            c6, c7 = st.columns(2)
            with c6:
                hs_text = st.text_input('Hs values (m)', value='1.5, 2.0')
            with c7:
                tp_text = st.text_input('Tp values (s)', value='8, 9, 10, 11, 12, 13')

            if st.button('Calculate sea-state grid'):
                hs_values = parse_range(hs_text)
                tp_values = parse_range(tp_text)
                rows = []
                for hs in hs_values:
                    for tp in tp_values:
                        result = compute_response(omega, rao_point, hs, tp, gamma, duration_hours)
                        rows.append({
                            'sheet': sheet_name,
                            'x_m': x_point,
                            'y_m': y_point,
                            'z_m': z_point,
                            'Hs_m': hs,
                            'Tp_s': tp,
                            'gamma': gamma,
                            'duration_h': duration_hours,
                            **result,
                        })
                grid_df = pd.DataFrame(rows)
                numeric_cols = [c for c in grid_df.columns if c not in ['sheet']]
                grid_df[numeric_cols] = grid_df[numeric_cols].round(4)
                st.session_state.grid_result_df = grid_df
                st.session_state.single_result_df = None

            if st.session_state.grid_result_df is not None:
                st.subheader('Grid results')
                st.dataframe(st.session_state.grid_result_df, use_container_width=True)

                plot_metric = st.selectbox('Plot metric', [
                    'RMS vertical displacement (m)',
                    'Most probable max vertical displacement (m)',
                    'RMS vertical velocity (m/s)',
                    'Most probable max vertical velocity (m/s)',
                    'RMS vertical acceleration (m/s²)',
                ], key='plot_metric')
                pivot = st.session_state.grid_result_df.pivot(index='Tp_s', columns='Hs_m', values=plot_metric).sort_index()
                st.line_chart(pivot)

    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.info('Upload your RAO Excel workbook to begin.')
