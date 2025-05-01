use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Zipf, Pareto, Normal, Distribution};
use plotters::prelude::*;
use plotters::style::{ShapeStyle};
use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};
use std::cmp::Ordering;
use std::fs::File;
use std::io::Write as IoWrite;

fn ks_test_uniform(sorted_sample: &[f64]) -> f64 {
    let n = sorted_sample.len() as f64;
    let mut d: f64 = 0.0;
    for (i, &x) in sorted_sample.iter().enumerate() {
        let empirical_cdf = (i as f64 + 1.0) / n;
        let lower_bound = i as f64 / n;
        let diff_upper = (empirical_cdf - x).abs();
        let diff_lower = (x - lower_bound).abs();
        d = d.max(diff_upper).max(diff_lower);
    }
    d
}

fn ks_test_no_sort(sample: &[f64]) -> f64 {
    let n = sample.len() as f64;
    let mut d: f64 = 0.0;
    for (i, &x) in sample.iter().enumerate() {
        let empirical_cdf = (i as f64 + 1.0) / n;
        let lower_bound = i as f64 / n;
        let diff_upper = (empirical_cdf - x).abs();
        let diff_lower = (x - lower_bound).abs();
        d = d.max(diff_upper).max(diff_lower);
    }
    d
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // 用户选择随机数生成方式。
    print!("Enter '1' to choose Mersenne Twister (MT19937), '2' for non-uniform distribution, or '3' for standard normal distribution: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let rng_choice: i32 = input.trim().parse().unwrap_or(1);

    // 用户选择种子方式。
    print!("Enter '1' for random seed, or '2' for fixed seed (5489): ");
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let seed_choice: i32 = input.trim().parse().unwrap_or(1);

    // 初始化随机数生成器。
    let mut rng_mt = if seed_choice == 2 {
        StdRng::seed_from_u64(5489)
    } else {
        StdRng::from_entropy()
    };

    let mut rng_zipf = if seed_choice == 2 {
        StdRng::seed_from_u64(5489)
    } else {
        StdRng::from_entropy()
    };

    let mut rng_pareto = if seed_choice == 2 {
        StdRng::seed_from_u64(5489)
    } else {
        StdRng::from_entropy()
    };

    let mut rng_normal = if seed_choice == 2 {
        StdRng::seed_from_u64(5489)
    } else {
        StdRng::from_entropy()
    };

    // 定义 Zipf, Pareto 和 Normal 分布。
    let zipf = Zipf::new(10000, 1.1)?;
    let pareto = Pareto::new(1.0, 3.0)?;
    let normal = Normal::new(0.0, 1.0)?;

    // 用户选择是否进入批量生成模式。
    print!("Enter '1' for normal mode or '2' for batch generation mode (no plot, output to file): ");
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let mode_choice: i32 = input.trim().parse().unwrap_or(1);

    let randomizer: usize = if mode_choice == 2 {
        // 批量模式输入范围
        print!("Enter the sample range to generate (e.g., from 0 to 10000000): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let range: usize = input.trim().parse().unwrap_or(10000000);
        range
    } else {
        // 普通模式输入样本数量
        print!("Enter the number of random numbers to generate: ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        input.trim().parse().unwrap_or(1000)
    };

    let mut data = Vec::with_capacity(randomizer);
    let mut random_numbers = Vec::with_capacity(randomizer);
    let mut sum_angles = 0.0;

    // 生成随机数和计算角度
    let gen_start = Instant::now();
    for _ in 0..randomizer {
        let rnd_x = match rng_choice {
            1 => rng_mt.gen_range(1.0..randomizer as f64), // Mersenne Twister
            2 => zipf.sample(&mut rng_zipf) as f64, // Zipf
            3 => normal.sample(&mut rng_normal), // Normal distribution
            _ => rng_mt.gen_range(1.0..randomizer as f64),
        };

        let rnd_y = match rng_choice {
            1 => rng_mt.gen_range(1.0..randomizer as f64),
            2 => pareto.sample(&mut rng_pareto),
            3 => normal.sample(&mut rng_normal),
            _ => rng_mt.gen_range(1.0..randomizer as f64),
        };

        let theta = rnd_y.atan2(rnd_x);
        data.push((rnd_x, rnd_y));
        random_numbers.push(rnd_x);
        sum_angles += theta;
    }

    let gen_duration = gen_start.elapsed();
    println!("Random number generation time: {:?}", gen_duration);

    // 角度累加与平均计算
    let angle_start = Instant::now();
    let avg_angle = sum_angles / randomizer as f64;
    let angle_duration = angle_start.elapsed();
    println!("Angle accumulation and average calculation time: {:?}", angle_duration);
    println!("Average Angle: {:.5} degrees", avg_angle.to_degrees());

    // 计算 KS 测试统计量（排序）
    let ks_sort_start = Instant::now();
    let mut normalized: Vec<f64> = random_numbers.iter().map(|&x| x / randomizer as f64).collect();
    normalized.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let ks_statistic_sorted = ks_test_uniform(&normalized);
    let ks_sort_duration = ks_sort_start.elapsed();
    println!("KS Test (with sorting) Statistic: {:.5}", ks_statistic_sorted);
    println!("KS Test (with sorting) computation time: {:?}", ks_sort_duration);

    // 计算 KS 测试统计量（无排序）
    let ks_no_sort_start = Instant::now();
    let ks_statistic_no_sort = ks_test_no_sort(&random_numbers);
    let ks_no_sort_duration = ks_no_sort_start.elapsed();
    println!("KS Test (no sort) Statistic: {:.5}", ks_statistic_no_sort);
    println!("KS Test (no sort) computation time: {:?}", ks_no_sort_duration);


    // 如果是批量模式，输出到文件而非绘图
    if mode_choice == 2 {
        if rng_choice != 1 {
            eprintln!("Error: Batch mode is only supported for uniform distribution.");
            return Ok(());
        }

        let file = File::create("ks_statistics.txt")?;
        let mut writer = io::BufWriter::new(file);

        for n in (10_000..=randomizer).step_by(10_000) {
            let mut avg_list = Vec::with_capacity(30);

            for _ in 0..30 {
                // 强制使用随机种子（seed_choice = 1）
                let mut rng = StdRng::from_entropy();

                let mut sum_theta = 0.0;

                for _ in 0..n {
                    let x = rng.gen_range(-1.0..=1.0);
                    let y:f64 = rng.gen_range(-1.0..=1.0);
                    let theta = y.atan2(x);
                    sum_theta += theta;
                }

                let avg_theta = sum_theta / n as f64;
                avg_list.push(avg_theta);
            }

            // 每完成30轮后计算一次方差
            let mean = avg_list.iter().sum::<f64>() / avg_list.len() as f64;
            let variance = avg_list.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / avg_list.len() as f64;

            writeln!(writer, "Samples: {} - Thetā Variance: {:.10}", n, variance)?;
        }

        println!("Batch mode completed. Results saved to ks_statistics.txt.");

    } else {

    let x_min_dynamic = data.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max_dynamic = data.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min_dynamic = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max_dynamic = data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_min_dynamic - 0.05)..(x_max_dynamic + 0.05);
        let y_range = (y_min_dynamic - 0.05)..(y_max_dynamic + 0.05);

        // Create and set up the drawing area for the scatter plot with 8K resolution (300 DPI).
        let dpi = 300;
        let width_in_inches = 25.6; // 8K resolution at 300 DPI
        let height_in_inches = 14.4; // 8K resolution at 300 DPI
        let width_in_pixels = (width_in_inches * dpi as f64).round() as u32;
        let height_in_pixels = (height_in_inches * dpi as f64).round() as u32;

        let root = BitMapBackend::new("scatter_non_uniform_8k.png", (width_in_pixels, height_in_pixels))
            .into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Scatter Plot", ("Arial", 50))
            .build_cartesian_2d(x_range, y_range)?;

        // Draw scatter plot with colorblind-friendly color scheme (Blue and Orange)
        chart.draw_series(
            data.iter().cloned().map(|(x, y)| Circle::new((x, y), 2, ShapeStyle {
                color: RGBColor(0, 0, 255).into(),  // Blue color
                filled: true,
                stroke_width: 1,
            }))
        )?
            .label("Random Points")
            .legend(|(x, y)| Circle::new((x, y), 2, ShapeStyle {
                color: RGBColor(0, 0, 255).into(),  // Blue color
                filled: true,
                stroke_width: 1,
            }));

        // Draw average angle line with contrasting color (Orange)
        let scale = 50.0;
        let avg_angle_x = x_min_dynamic + avg_angle.cos() * scale;
        let avg_angle_y = y_min_dynamic + avg_angle.sin() * scale;
        chart.draw_series(LineSeries::new(
            vec![(x_min_dynamic, y_min_dynamic), (avg_angle_x, avg_angle_y)],
            ShapeStyle { color: RGBColor(255, 165, 0).into(), filled: true, stroke_width: 3 },  // Orange color
        ))?;

        // Display the mesh/grid lines.
        chart.configure_mesh().x_labels(30).y_labels(30).draw()?;

        root.present()?;
        println!("Scatter plot generated successfully!");
    }

    println!("Total execution time: {:?}", start_time.elapsed());
    Ok(())
}
