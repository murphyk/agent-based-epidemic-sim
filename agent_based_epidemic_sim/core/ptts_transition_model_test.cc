// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "agent_based_epidemic_sim/core/ptts_transition_model.h"

#include "absl/time/time.h"
#include "agent_based_epidemic_sim/core/visit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace abesim {
namespace {

TEST(PTTSTransitionModelTest, UpdatesTransitionModel) {
  PTTSTransitionModel::StateTransitionDiagram state_transition_diagram{{
      {
          {.transitions = absl::discrete_distribution<int>({0.5, 0.5, 0, 0}),
           .rate = 1},
          {.transitions = absl::discrete_distribution<int>({0, 0, 0.8, 0.2}),
           .rate = .5},
          {.transitions = absl::discrete_distribution<int>({0, 0, 0, 1}),
           .rate = .1},
          {.transitions = absl::discrete_distribution<int>({0, 0, 0, 1}),
           .rate = 1},
      },
  }};
  PTTSTransitionModel model(state_transition_diagram);
  std::vector<HealthTransition> health_transitions;
  health_transitions.push_back({.time = absl::FromUnixSeconds(0LL),
                                .health_state = HealthState::SUSCEPTIBLE});
  for (int i = 0; i < 10; ++i) {
    health_transitions.push_back(
        model.GetNextHealthTransition(health_transitions[i]));
  }
  EXPECT_EQ(HealthState::RECOVERED, health_transitions.rbegin()->health_state);
}

}  // namespace
}  // namespace abesim
