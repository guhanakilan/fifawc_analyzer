import React from "react";

const THEME_KEY = "nova-enhancer:theme";

export const ThemeCtx = React.createContext({ theme: "light", toggle: () => {} });
export const useTheme = () => React.useContext(ThemeCtx);

export function ThemeProvider({ children }) {
  const [theme, setTheme] = React.useState(() => {
    try {
      return localStorage.getItem(THEME_KEY) || "light";
    } catch {
      return "light";
    }
  });

  React.useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage disabled — the theme still applies for this session */
    }
  }, [theme]);

  const value = React.useMemo(
    () => ({ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }),
    [theme],
  );
  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}
