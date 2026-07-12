# TonightSky v1.0

Welcome to **TonightSky**, your companion for planning astronomical observations. This guide will help you navigate the app's features and use it effectively.

---

## **Introduction**
TonightSky helps users calculate Local Sidereal Time (LST), list astronomical objects based on specified criteria, and visualize altitude graphs for observing targets.

Storage model: user settings and profile/config data are stored in the browser (localStorage/sessionStorage), not in a server-side user database. This avoids needing account/user-management software and dedicated per-user server storage.

---

## **Features**
- **Settings Configuration**:
  - **Latitude & Longitude**: Specify your observation location's coordinates.
  - **Timezone**: Choose your local timezone for accurate calculations.
  - **Date and Local Time**: Enter the desired date and time for observations.
  - **Filter Query**: Perform SQL-like queries on the object list.
  - **Catalog Selection**: Choose catalogs (e.g., Messier, NGC, IC) for your object search.

- **Object List**:
  - View astronomical objects matching your criteria in a sortable table.
  - Double-click a row to search for the object on Astrobin.
  - Right-click or click on the "Altitude" cell to display a graph of the object's altitude.

- **Altitude Graph**:
  - Visualize the altitude of an object over time.
  - Key events such as sunset, sunrise, and transit are marked.
  - The visible period is highlighted in green.

- **Tonight's Best Ranking**:
  - Ranks the already-filtered object list for imaging suitability instead of only returning catalog order.
  - Requires a selected equipment profile (focal length, aperture, sensor width and height).
  - Uses an optional selected horizon profile; if none is selected, the server default horizon is used.
  - Adds two result columns: **Score** and **Shooting Time**.

---

## **How to Use**

### **1. Configure Settings**
1. Enter your **latitude** and **longitude**.
2. Choose your **timezone**.
3. Set the **date** and **local time**.

### **2. Apply Filters**
- Use the **Filter Query** field to narrow down objects. Example queries:
  - `altitude > 50 and relative tt < 03 and direction = south`
  - `magnitude < 5 and type = galaxy`
  - `transit time > '21:00' and altitude > 30`
  - `catalog = messier and magnitude < 6`

### **3. List Objects**
- Click **List Objects** to display objects that match your criteria in a table.

### **4. Interact with the Table**
- Click on column headers to sort the table.
- Right-click or click on an "Altitude" cell to open the graph modal.
- Double-click a row to open its Astrobin page.

### **5. View Altitude Graph**
- The graph shows:
  - **Transit Time**: The moment the object is at its highest altitude.
  - **Visible Period**: Highlighted in green, indicating when the object is above the horizon.
  - **Sunset, Sunrise, and Other Events**: Marked with vertical lines.

### **6. Use Tonight's Best**
1. Select or create an **Equipment Profile**.
2. Optionally select a **Horizon Profile**.
3. Set your normal filters (catalog checkboxes and filter query).
4. Click **Tonight's Best**.

The app sends a normal `/api/list_objects` request with `tonights_best: true`, plus:
- `equipment`: focal length, aperture, sensor width, sensor height.
- `horizon_points` (optional): selected custom horizon profile points.

The server then:
1. Builds the normal eligible set first (catalog selection, query filter, and `altitude >= 0` at requested local time).
2. Computes tonight's astronomical dusk and dawn for the observer location.
3. Scores each eligible object.
4. Returns the same rows, re-ordered by descending composite score, with `Score` and `Shooting Time` populated.

---

## **Tonight's Best Scoring Logic**

Tonight's Best uses a composite score with one important design choice: **size fit is a gate multiplier**.

Composite score:

`composite = size_score * (0.5 * brightness_score + (1/3) * time_score + (1/6) * moon_score)`

### **1. Size Score (Gate)**
- Parses `Size` as either a single value (`12.9`) or `WxH` (`6.0x4.0`) in arcminutes.
- Uses the object's major axis against the limiting frame dimension (`min(fov_width, fov_height)`).
- Piecewise behavior:
  - If object does not fit (`fill_fraction > 1.0`): score `0.0`.
  - If underfilled (`fill_fraction < 0.4`): score `(fill_fraction / 0.4)^2`.
  - If well framed (`0.4 <= fill_fraction <= 0.8`): score `1.0`.
  - If oversized but still fitting (`0.8 < fill_fraction <= 1.0`): linearly tapers back to `0.0`.
- Unknown or unparsable size uses type-aware fallback:
  - nebula: `0.35`
  - cluster: `0.20`
  - planetary nebula: `0.10`
  - galaxy: `0.05`
  - other/unknown type: `0.05`

### **2. Brightness Score**
- Magnitude normalized to `[0, 1]` between:
  - `MAG_BRIGHT = 4.0` -> score `1.0`
  - `MAG_FAINT = 14.0` -> score `0.0`
- Unknown magnitude defaults to `0.5`.

### **3. Time Score (Best Shooting Window)**
- Builds sample times from astronomical dusk to dawn.
- Uses `SAMPLE_INTERVAL_MINUTES = 10` by default.
- Samples are scanned for the longest continuous run where:
  - `object_altitude(t) > horizon_min_altitude(azimuth(t))`
- `Shooting Time` is this best run length in minutes (displayed as HH:MM style).
- Time score is:
  - `time_fraction = run_minutes / total_darkness_minutes` (clamped to `<= 1.0`)
  - plus small tiebreak term: `0.001 * (peak_clearance_deg / 90)`

### **4. Moon Score**
- If Moon is below horizon: `1.0`.
- Otherwise:
  - `separation_fraction = separation_deg / 180`
  - `penalty = moon_illumination * (1 - separation_fraction)`
  - `moon_score = 1 - penalty`

### **Sampling Notes**
- A 10-minute interval is a performance/precision tradeoff.
- Short near-horizon windows can be slightly undercounted at coarse sampling.
- Moving to 5-minute sampling improves precision for short windows at about 2x coordinate-transform cost.

---

## **Keyboard Shortcuts**
- Press **Enter** to list objects.
- Click the **Help (?)** button for additional guidance.

---

## **Tips & Notes**
- Large catalogs (e.g., 10,000 objects) may take 60+ seconds to load.
- Objects with negative altitude at the specified time are excluded.
- Double-clicking a row opens the object's Astrobin page.

---

## **Screenshots**
*Add relevant screenshots here to demonstrate the features.*

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/mpinnuck/webTonightSky.git

## **Issues**
- query OR logic todo.
