'use client'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          DAO와 거버넌스
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            DAO(Decentralized Autonomous Organization)는 스마트 컨트랙트로 운영되는
            탈중앙화 자율 조직으로, 커뮤니티가 직접 의사결정에 참여합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🏛️ DAO 구조와 운영
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            거버넌스 스마트 컨트랙트
          </h4>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`contract GovernorDAO {
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startBlock;
        uint256 endBlock;
        bool executed;
        mapping(address => bool) hasVoted;
    }
    
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    uint256 public quorum = 4%; // 4% of total supply
    uint256 public votingPeriod = 3 days;
    
    function propose(string memory description) external returns (uint256) {
        require(getVotingPower(msg.sender) >= proposalThreshold, "Insufficient voting power");
        
        proposalCount++;
        Proposal storage newProposal = proposals[proposalCount];
        newProposal.id = proposalCount;
        newProposal.proposer = msg.sender;
        newProposal.description = description;
        newProposal.startBlock = block.number;
        newProposal.endBlock = block.number + votingPeriod;
        
        emit ProposalCreated(proposalCount, msg.sender, description);
        return proposalCount;
    }
    
    function vote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];
        require(block.number <= proposal.endBlock, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        
        uint256 votingPower = getVotingPower(msg.sender);
        proposal.hasVoted[msg.sender] = true;
        
        if (support) {
            proposal.forVotes += votingPower;
        } else {
            proposal.againstVotes += votingPower;
        }
        
        emit VoteCast(msg.sender, proposalId, support, votingPower);
    }
}`}
            </pre>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🗳️ 투표 메커니즘
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">투표 방식</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h5 className="font-semibold text-sm mb-1">Token Voting</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  1 토큰 = 1 투표권
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h5 className="font-semibold text-sm mb-1">Quadratic Voting</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  투표 비용 = 투표수²
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h5 className="font-semibold text-sm mb-1">Delegation</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  투표권 위임 가능
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">제안 프로세스</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">1</div>
                <span className="text-sm">제안 제출 (최소 토큰 필요)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">2</div>
                <span className="text-sm">토론 기간</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">3</div>
                <span className="text-sm">투표 기간</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">4</div>
                <span className="text-sm">타임락 (보안)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">5</div>
                <span className="text-sm">실행</span>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💼 Treasury 관리
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            DAO Treasury 운영
          </h4>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">수입원</h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 프로토콜 수수료</li>
                <li>• NFT 판매</li>
                <li>• 투자 수익</li>
                <li>• 기부금</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">지출 항목</h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 개발자 보상</li>
                <li>• 마케팅 비용</li>
                <li>• 감사 비용</li>
                <li>• 그랜트 프로그램</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">관리 도구</h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Gnosis Safe</li>
                <li>• Snapshot</li>
                <li>• Tally</li>
                <li>• Boardroom</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}