# TonightSky v1.0

Welcome to **TonightSky**, your companion for planning astronomical observations. This guide will help you navigate the app's features and use it effectively.

---

## **Introduction**
TonightSky helps users calculate Local Sidereal Time (LST), list astronomical objects based on specified criteria, and visualize altitude graphs for observing targets.

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
