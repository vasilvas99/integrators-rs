use std::error::Error;

use clap::Parser;
use gnuplot::{
    AutoOption, AxesCommon, Caption, Color, Figure,
    PlotOption::{LineStyle, LineWidth},
};
use integrators_rs::{euler::PredictorCorrector12, IVPIntegrator, IVProblem, StaticVector};
use ndarray::Array;

static GRAVITY: f64 = 9.81; // m*s^-2

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 0.0)]
    /// Initial time (s)
    t0: f64,

    #[arg(long, default_value_t = 50.0)]
    /// Initial velocity (m/s)
    v0: f64,

    #[arg(long, default_value_t = std::f64::consts::PI/4.0)]
    /// Shooting angle (rad)
    theta0: f64,

    #[arg(long, default_value_t = 0.0)]
    /// Starting x position (m)
    x0: f64,

    #[arg(long, default_value_t = 0.0)]
    /// Starting y position (m)
    y0: f64,

    #[arg(long, default_value_t = 10.0)]
    /// Object mass (kg)
    object_mass: f64,

    #[arg(long, default_value_t = 0.2)]
    /// Air drag coefficient (dimensionless)
    air_drag: f64,

    #[arg(long, default_value_t = 1.29)]
    /// Air density (kg*m^-3)
    air_density: f64,

    #[arg(long, default_value_t = 0.25)]
    /// Cross section area (m^2)
    cross_section: f64,

    #[arg(long, default_value_t = 0.01)]
    /// Time step for integration
    time_step: f64,
}

fn rhs_params(
    _t: f64,
    u: &StaticVector<f64>,
    m: f64,
    c: f64,
    s: f64,
    rho: f64,
) -> StaticVector<f64> {
    assert_eq!(u.len(), 4, "The ODE system has 4 equations");

    let _x = u[0];
    let _y = u[1];
    let v = u[2];
    let theta = u[3];

    let res = vec![
        v * theta.cos(),
        v * theta.sin(),
        -(1.0 / (2.0 * m)) * c * rho * s * v * v - GRAVITY * theta.sin(),
        -(GRAVITY / v) * theta.cos(),
    ];
    Array::from_vec(res)
}

fn plot(x_vals: Vec<f64>, y_vals: Vec<f64>, params: &Args) -> Result<(), Box<dyn Error>> {
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(
            &format!(
                "Mass: {m:.3} kg, Drag coefficient: {c:.3}, v_0: {v0:.3} m/s\n
                Cross section: {s:.3} m^2, Air density: {rho:.3} kg/m^{{3}}, Shooting angle: {theta:.3} rad",
                m = params.object_mass,
                c = params.air_drag,
                v0 = params.v0,
                s = params.cross_section,
                rho = params.air_density,
                theta = params.theta0
            ),
            &[],
        )
        .lines(
            &x_vals,
            &y_vals,
            &[
                LineWidth(2.),
                Color("black"),
                LineStyle(gnuplot::Solid),
                Caption(&format!("Rocket trajectory")),
            ],
        )
        .set_y_range(AutoOption::Fix(0.0), AutoOption::Auto);
    fg.show_and_keep_running()?;
    Ok(())
}

fn main() {
    let cli = Args::parse();
    let y0 = Array::from_vec(vec![cli.x0, cli.y0, cli.v0, cli.theta0]);
    let mut problem = IVProblem::new(cli.t0, y0, move |t, y| {
        rhs_params(
            t,
            y,
            cli.object_mass,
            cli.air_drag,
            cli.cross_section,
            cli.air_density,
        )
    });

    let step = cli.time_step;
    let mut t = cli.t0;
    let integrator = PredictorCorrector12::new(step, Some(10));

    let mut x_vals = vec![cli.x0];
    let mut y_vals = vec![cli.y0];

    while problem.y[1] >= 0.0 {
        t += step;
        integrator.step_until(&mut problem, t);
        if problem.t - t > 5.0 * step {
            panic!("Integrator failed to step ahead 5 times => Integration failed.")
        }
        x_vals.push(problem.y[0]);
        y_vals.push(problem.y[1]);
    }
    println!("{:?}", cli);
    plot(x_vals, y_vals, &cli).expect("Plotting failed");
}
