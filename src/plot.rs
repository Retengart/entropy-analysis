use std::path::Path;

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::model::LetterRow;

pub fn plot_histogram(path: &Path, rows: &[LetterRow], sorted: bool) -> Result<()> {
    let max_p = rows.iter().map(|r| r.p).fold(0.0f64, f64::max);
    let n = rows.len();
    let caption = if sorted { "p_i (sorted)" } else { "p_i by alphabet" };

    let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
    } else {
        (false, path.to_string_lossy().to_string())
    };

    if is_svg {
        let root = SVGBackend::new(&path_str, (1200, 600)).into_drawing_area();
        draw_histogram(&root, rows, max_p, n, caption)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(&path_str, (1200, 600)).into_drawing_area();
        draw_histogram(&root, rows, max_p, n, caption)?;
        root.present()?;
    }
    Ok(())
}

fn draw_histogram<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    rows: &[LetterRow],
    max_p: f64,
    n: usize,
    caption: &str,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..(n as i32), 0.0f64..(max_p * 1.1 + 1e-9))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(n)
        .x_label_formatter(&|v| {
            let idx = (*v as usize).saturating_sub(1).min(n.saturating_sub(1));
            if *v == 0 { String::new() } else { rows[idx].letter.to_string() }
        })
        .y_desc("p_i")
        .x_desc("n")
        .draw()?;

    chart.draw_series(rows.iter().enumerate().map(|(i, r)| {
        let rect = Rectangle::new([(i as i32, 0.0), ((i as i32) + 1, r.p)], BLUE.mix(0.6).filled());
        rect
    }))?;
    Ok(())
}


