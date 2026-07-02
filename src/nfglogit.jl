struct MixedStrategyProfile where N
    dims::NTuple{N,Int}
    weights::
end

void PointToProfile(MixedStrategyProfile<double> &p_profile, const Vector<double> &p_point)
{
  for (size_t i = 1; i < p_point.size(); i++) {
    p_profile[i] = std::exp(p_point[i]);
  }
}

void PointToLogProfile(MixedStrategyProfile<double> &p_profile, const Vector<double> &p_point)
{
  for (size_t i = 1; i < p_point.size(); i++) {
    p_profile[i] = p_point[i];
  }
}

Vector<double> ProfileToPoint(const LogitQREMixedStrategyProfile &p_profile)
{
  Vector<double> point(p_profile.size() + 1);
  for (size_t i = 1; i <= p_profile.size(); i++) {
    point[i] = std::log(p_profile[i]);
  }
  point.back() = p_profile.GetLambda();
  return point;
}

double LogLike(const Vector<double> &p_frequencies, const Vector<double> &p_point)
{
  return std::inner_product(
      p_frequencies.begin(), p_frequencies.end(), p_point.begin(), 0.0, std::plus<>(),
      [](const double freq, const double prob) { return freq * std::log(prob); });
}

double DiffLogLike(const Vector<double> &p_frequencies, const Vector<double> &p_tangent)
{
  return std::inner_product(p_frequencies.begin(), p_frequencies.end(), p_tangent.begin(), 0.0);
}

bool RegretTerminationFunction(const Game &p_game, const Vector<double> &p_point, double p_regret)
{
  if (p_point.back() < 0.0) {
    return true;
  }
  MixedStrategyProfile<double> profile(p_game->NewMixedStrategyProfile(0.0));
  PointToProfile(profile, p_point);
  return profile.GetMaxRegret() < p_regret;
}

class Equation {
public:
  virtual ~Equation() = default;

  virtual double Value(const MixedStrategyProfile<double> &p_profile,
                       const MixedStrategyProfile<double> &p_logProfile,
                       const Vector<double> &p_strategyValues, double p_lambda) const = 0;
  virtual void Gradient(const MixedStrategyProfile<double> &p_profile,
                        const Vector<double> &p_strategyValues,
                        const Matrix<double> &p_strategyDerivs, double p_lambda,
                        Vector<double> &p_gradient) const = 0;
};

class SumToOneEquation final : public Equation {
  Game m_game;
  GamePlayer m_player;
  int m_firstIndex{0}, m_lastIndex{0};

public:
  explicit SumToOneEquation(const GamePlayer &p_player)
    : m_game(p_player->GetGame()), m_player(p_player)
  {
    int col = 1;
    for (const auto &player : m_game->GetPlayers()) {
      if (player != m_player) {
        col += player->GetStrategies().size();
        continue;
      }
      m_firstIndex = col;
      m_lastIndex = col + player->GetStrategies().size();
      return;
    }
  }

  ~SumToOneEquation() override = default;
  double Value(const MixedStrategyProfile<double> &p_profile,
               const MixedStrategyProfile<double> &p_logProfile,
               const Vector<double> &p_strategyValues, double p_lambda) const override;
  void Gradient(const MixedStrategyProfile<double> &p_profile,
                const Vector<double> &p_strategyValues, const Matrix<double> &p_strategyDerivs,
                double p_lambda, Vector<double> &p_gradient) const override;
};

double SumToOneEquation::Value(const MixedStrategyProfile<double> &p_profile,
                               const MixedStrategyProfile<double> &p_logProfile,
                               const Vector<double> &p_strategyValues, double p_lambda) const
{
  double value = -1.0;
  for (int col = m_firstIndex; col < m_lastIndex; col++) {
    value += p_profile[col];
  }
  return value;
}

void SumToOneEquation::Gradient(const MixedStrategyProfile<double> &p_profile,
                                const Vector<double> &p_strategyValues,
                                const Matrix<double> &p_strategyDerivs, double p_lambda,
                                Vector<double> &p_gradient) const
{
  p_gradient = 0.0;
  for (int col = m_firstIndex; col < m_lastIndex; col++) {
    p_gradient[col] = p_profile[col];
  }
}

class RatioEquation final : public Equation {
  Game m_game;
  GamePlayer m_player;
  GameStrategy m_strategy, m_refStrategy;
  int m_strategyIndex{0}, m_refStrategyIndex{0};

public:
  RatioEquation(const GameStrategy &p_strategy, const GameStrategy &p_refStrategy)
    : m_game(p_strategy->GetGame()), m_player(p_strategy->GetPlayer()), m_strategy(p_strategy),
      m_refStrategy(p_refStrategy)
  {
    int col = 1;
    for (const auto &strategy : m_game->GetStrategies()) {
      if (strategy == p_strategy) {
        m_strategyIndex = col;
      }
      else if (strategy == p_refStrategy) {
        m_refStrategyIndex = col;
      }
      col++;
    }
  }

  ~RatioEquation() override = default;
  double Value(const MixedStrategyProfile<double> &p_profile,
               const MixedStrategyProfile<double> &p_logProfile,
               const Vector<double> &p_strategyValues, double p_lambda) const override;
  void Gradient(const MixedStrategyProfile<double> &p_profile,
                const Vector<double> &p_strategyValues, const Matrix<double> &p_strategyDerivs,
                double p_lambda, Vector<double> &p_gradient) const override;
};

double RatioEquation::Value(const MixedStrategyProfile<double> &p_profile,
                            const MixedStrategyProfile<double> &p_logProfile,
                            const Vector<double> &p_strategyValues, double p_lambda) const
{
  return p_logProfile[m_strategyIndex] - p_logProfile[m_refStrategyIndex] -
         p_lambda * (p_strategyValues[m_strategyIndex] - p_strategyValues[m_refStrategyIndex]);
}

void RatioEquation::Gradient(const MixedStrategyProfile<double> &p_profile,
                             const Vector<double> &p_strategyValues,
                             const Matrix<double> &p_strategyDerivs, double p_lambda,
                             Vector<double> &p_gradient) const
{
  int col = 1;
  for (const auto &player : m_game->GetPlayers()) {
    for (const auto &strategy : player->GetStrategies()) {
      if (strategy == m_refStrategy) {
        p_gradient[col] = -1.0;
      }
      else if (strategy == m_strategy) {
        p_gradient[col] = 1.0;
      }
      else if (player == m_player) {
        p_gradient[col] = 0.0;
      }
      else {
        p_gradient[col] =
            -p_lambda * p_profile[col] *
            (p_strategyDerivs(m_strategyIndex, col) - p_strategyDerivs(m_refStrategyIndex, col));
      }
      col++;
    }
  }
  p_gradient[col] = p_strategyValues[m_refStrategyIndex] - p_strategyValues[m_strategyIndex];
}

class EquationSystem {
public:
  explicit EquationSystem(const Game &p_game);
  ~EquationSystem() = default;

  void GetValue(const Vector<double> &p_point, Vector<double> &p_lhs) const;
  void GetJacobian(const Vector<double> &p_point, Matrix<double> &p_jac) const;

private:
  std::vector<std::shared_ptr<Equation>> m_equations;
  const Game &m_game;
  mutable MixedStrategyProfile<double> m_profile, m_logProfile;
  mutable Vector<double> m_strategyValues;
  mutable Matrix<double> m_strategyDerivs;
};

EquationSystem::EquationSystem(const Game &p_game)
  : m_game(p_game), m_profile(p_game->NewMixedStrategyProfile(0.0)),
    m_logProfile(p_game->NewMixedStrategyProfile(0.0)),
    m_strategyValues(m_profile.MixedProfileLength()),
    m_strategyDerivs(m_profile.MixedProfileLength(), m_profile.MixedProfileLength())
{
  m_equations.reserve(m_profile.MixedProfileLength());
  for (const auto &player : m_game->GetPlayers()) {
    m_equations.push_back(std::make_shared<SumToOneEquation>(player));
    auto strategies = player->GetStrategies();
    for (auto strategy = std::next(strategies.begin()); strategy != strategies.end(); ++strategy) {
      m_equations.push_back(std::make_shared<RatioEquation>(*strategy, strategies.front()));
    }
  }
}

void EquationSystem::GetValue(const Vector<double> &p_point, Vector<double> &p_lhs) const
{
  PointToProfile(m_profile, p_point);
  PointToLogProfile(m_logProfile, p_point);
  const double lambda = p_point.back();
  int col = 1;
  for (const auto &strategy : m_game->GetStrategies()) {
    m_strategyValues[col++] = m_profile.GetPayoff(strategy);
  }
  std::transform(m_equations.begin(), m_equations.end(), p_lhs.begin(), [this, lambda](auto e) {
    return e->Value(m_profile, m_logProfile, m_strategyValues, lambda);
  });
}

void EquationSystem::GetJacobian(const Vector<double> &p_point, Matrix<double> &p_jac) const
{
  PointToProfile(m_profile, p_point);
  const double lambda = p_point.back();
  Vector<double> column(p_point.size());
  m_strategyDerivs = 0.0;
  int row = 1;
  for (const auto &strategy1 : m_game->GetStrategies()) {
    m_strategyValues[row] = m_profile.GetPayoff(strategy1);
    int col = 1;
    for (const auto &strategy2 : m_game->GetStrategies()) {
      if (strategy1->GetPlayer() != strategy2->GetPlayer()) {
        m_strategyDerivs(row, col) =
            m_profile.GetPayoffDeriv(strategy1->GetPlayer()->GetNumber(), strategy1, strategy2);
      }
      col++;
    }
    row++;
  }
  for (size_t i = 1; i <= m_equations.size(); i++) {
    m_equations[i - 1]->Gradient(m_profile, m_strategyValues, m_strategyDerivs, lambda, column);
    p_jac.SetColumn(i, column);
  }
}

class TracingCallbackFunction {
public:
  TracingCallbackFunction(const Game &p_game, const MixedStrategyObserverFunctionType &p_observer)
    : m_game(p_game), m_observer(p_observer)
  {
  }
  ~TracingCallbackFunction() = default;

  void AppendPoint(const Vector<double> &p_point);
  const std::list<LogitQREMixedStrategyProfile> &GetProfiles() const { return m_profiles; }

private:
  Game m_game;
  MixedStrategyObserverFunctionType m_observer;
  std::list<LogitQREMixedStrategyProfile> m_profiles;
};

void TracingCallbackFunction::AppendPoint(const Vector<double> &p_point)
{
  MixedStrategyProfile<double> profile(m_game->NewMixedStrategyProfile(0.0));
  PointToProfile(profile, p_point);
  m_profiles.emplace_back(profile, p_point.back(), 1.0);
  m_observer(m_profiles.back());
}



function LogitStrategySolve(p_start, p_regret::Float64, p_omega::Float64, p_firstStep::Float64, p_maxAccel::Float64, const MixedStrategyObserverFunctionType &p_observer)

  if (p_start.size() == 0)
    return p_start
  end

  if (const double scale = p_start.GetGame()->GetMaxPayoff() - p_start.GetGame()->GetMinPayoff();
      scale != 0.0)
    p_regret *= scale;
  end

  const Game game = p_start.GetGame();
  Vector<double> x(ProfileToPoint(p_start));
  TracingCallbackFunction callback(game, p_observer);
  EquationSystem system(game);
  tracer.TracePath(
      (p_point, p_lhs) -> GetValue(system, p_point, p_lhs),
      (p_point, p_jac) -> GetJacobian(system, p_point, p_jac),
      x,
      p_omega,
      p_point -> RegretTerminationFunction(p_start.GetGame(), p_point, p_regret),
      p_point -> callback.AppendPoint(p_point),
      p_maxAccel,
      p_firstStep);
  return callback.GetProfiles();
end

