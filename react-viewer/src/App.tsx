import React, { useState, useEffect } from 'react';

// Define the User interface
interface User {
  id: number;
  name: string;
  score: number;
}

const RealtimeLeaderboard: React.FC = () => {
  // State for leaderboard data
  const [users, setUsers] = useState<User[]>([]);
  
  // Generate mock data
  const generateMockUsers = (): User[] => {
    return [
      { id: 1, name: 'Alice', score: 1500 },
      { id: 2, name: 'Bob', score: 1200 },
      { id: 3, name: 'Charlie', score: 950 },
      { id: 4, name: 'David', score: 800 },
      { id: 5, name: 'Eve', score: 650 },
    ];
  };

  // Initialize data on mount
  useEffect(() => {
    const initialUsers = generateMockUsers();
    setUsers(initialUsers);
    
    // Update data every 5 seconds
    const interval = setInterval(() => {
      setUsers(prevUsers => {
        // Simulate random score changes
        return prevUsers.map(user => ({
          ...user,
          score: user.score + (Math.random() > 0.5 ? Math.floor(Math.random() * 100) : -Math.floor(Math.random() * 50))
        }));
      });
    }, 5000);
    
    // Cleanup interval on unmount
    return () => clearInterval(interval);
  }, []);

  // Sort users by score descending
  const sortedUsers = [...users].sort((a, b) => b.score - a.score);

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-6 text-center">Realtime Leaderboard</h1>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedUsers.map((user, index) => (
              <tr key={user.id} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {index + 1}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {user.name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {user.score.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {users.length === 0 && (
        <div className="mt-6 text-center text-gray-500">
          Loading leaderboard data...
        </div>
      )}
    </div>
  );
};

export default RealtimeLeaderboard;
