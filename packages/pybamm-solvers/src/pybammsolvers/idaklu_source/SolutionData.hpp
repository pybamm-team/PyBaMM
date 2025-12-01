#ifndef PYBAMM_IDAKLU_SOLUTION_DATA_HPP
#define PYBAMM_IDAKLU_SOLUTION_DATA_HPP

#include <memory>
#include "common.hpp"
#include "Solution.hpp"

/**
 * @brief SolutionData class. Contains all the data needed to create a Solution
 */
class SolutionData
{
  public:
    /**
     * @brief Default constructor
     */
    SolutionData() = default;

    /**
     * @brief Constructor using fields
     */
    SolutionData(
      int flag,
      int number_of_timesteps,
      int length_of_return_vector,
      int arg_sens0,
      int arg_sens1,
      int arg_sens2,
      int length_of_final_sv_slice,
      bool save_hermite,
      std::unique_ptr<sunrealtype[]> t_return,
      std::unique_ptr<sunrealtype[]> y_return,
      std::unique_ptr<sunrealtype[]> yp_return,
      std::unique_ptr<sunrealtype[]> yS_return,
      std::unique_ptr<sunrealtype[]> ypS_return,
      std::unique_ptr<sunrealtype[]> yterm_return)
      : flag(flag),
        number_of_timesteps(number_of_timesteps),
        length_of_return_vector(length_of_return_vector),
        arg_sens0(arg_sens0),
        arg_sens1(arg_sens1),
        arg_sens2(arg_sens2),
        length_of_final_sv_slice(length_of_final_sv_slice),
        save_hermite(save_hermite),
        t_return(std::move(t_return)),
        y_return(std::move(y_return)),
        yp_return(std::move(yp_return)),
        yS_return(std::move(yS_return)),
        ypS_return(std::move(ypS_return)),
        yterm_return(std::move(yterm_return))
    {}

    /**
     * @brief Destructor - unique_ptr handles cleanup automatically
     */
    ~SolutionData() = default;

    /**
     * @brief Deleted copy constructor
     */
    SolutionData(const SolutionData &solution_data) = delete;

    /**
     * @brief Deleted copy assignment
     */
    SolutionData& operator=(const SolutionData &solution_data) = delete;

    /**
     * @brief Move constructor - unique_ptr handles transfer automatically
     */
    SolutionData(SolutionData &&solution_data) noexcept = default;

    /**
     * @brief Move assignment - unique_ptr handles transfer automatically
     */
    SolutionData& operator=(SolutionData &&solution_data) noexcept = default;

    /**
     * @brief Create a solution object from this data
     */
    Solution generate_solution();

private:
    int flag = 0;
    int number_of_timesteps = 0;
    int length_of_return_vector = 0;
    int arg_sens0 = 0;
    int arg_sens1 = 0;
    int arg_sens2 = 0;
    int length_of_final_sv_slice = 0;
    bool save_hermite = false;
    std::unique_ptr<sunrealtype[]> t_return;
    std::unique_ptr<sunrealtype[]> y_return;
    std::unique_ptr<sunrealtype[]> yp_return;
    std::unique_ptr<sunrealtype[]> yS_return;
    std::unique_ptr<sunrealtype[]> ypS_return;
    std::unique_ptr<sunrealtype[]> yterm_return;
};

#endif // PYBAMM_IDAKLU_SOLUTION_DATA_HPP
